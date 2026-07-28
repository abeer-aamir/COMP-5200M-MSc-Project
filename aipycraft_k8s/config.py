from __future__ import annotations

import json
import ipaddress
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "benchmark" / "aipycraft_config.json"
FROZEN_API_BASE = "https://openrouter.ai/api/v1"
FROZEN_MODEL = "meta-llama/llama-3.2-1b-instruct"
FROZEN_PROVIDERS = ("cloudflare",)
FROZEN_INPUT_PRICE = Decimal("0.027")
FROZEN_OUTPUT_PRICE = Decimal("0.201")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    model: str
    provider_only: tuple[str, ...]
    allow_fallbacks: bool
    temperature: float
    transport_retries: int
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal


@dataclass(frozen=True)
class PipelineSettings:
    max_regenerations: int
    runtime_observation_seconds: int
    runtime_poll_seconds: float
    command_timeout_seconds: int
    output_root: Path


@dataclass(frozen=True)
class ImageLock:
    tag: str
    source: str
    digest: str


@dataclass(frozen=True)
class EnvironmentLock:
    path: Path
    kind_version: str
    kind_url: str
    kind_sha256: str
    kind_filename: str
    kubectl_version: str
    kubectl_url: str
    kubectl_sha256: str
    kubectl_filename: str
    kubernetes_minor: str
    node_image: str
    node_image_source: str
    node_image_id: str
    node_image_dockerfile: Path
    node_image_dockerfile_sha256: str
    node_image_entrypoint: Path
    node_image_entrypoint_sha256: str
    kind_config_sha256: str
    docker_network_subnet: str
    docker_network_gateway: str
    calico_version: str
    calico_manifest_url: str
    calico_manifest_sha256: str
    calico_filename: str
    preload_images: tuple[ImageLock, ...]


@dataclass(frozen=True)
class EnvironmentConfig:
    lock: EnvironmentLock
    kind_config: Path
    cache_root: Path


@dataclass(frozen=True)
class PromptConfig:
    generate: Path
    regenerate: Path


@dataclass(frozen=True)
class AppConfig:
    path: Path
    api: ApiConfig
    pipeline: PipelineSettings
    environment: EnvironmentConfig
    prompts: PromptConfig


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a JSON object in {path}")
    return value


def _project_path(value: Any, label: str, *, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty project-relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise ConfigError(f"{label} must be project-relative")
    resolved = (PROJECT_ROOT / raw).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ConfigError(f"{label} escapes the project root") from exc
    if must_exist and not resolved.exists():
        raise ConfigError(f"{label} does not exist: {resolved}")
    return resolved


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ConfigError(f"{label} must be a decimal number") from exc
    if result < 0:
        raise ConfigError(f"{label} cannot be negative")
    return result


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ConfigError(
            f"Invalid {label} fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def load_environment_lock(path: Path) -> EnvironmentLock:
    raw = _read_object(path)
    _exact_keys(
        raw,
        {
            "schema_version",
            "kind",
            "kubectl",
            "cluster",
            "calico",
            "preload_images",
        },
        "environment lock",
    )
    if raw["schema_version"] != 3:
        raise ConfigError("Unsupported environment lock schema_version")
    kind = _exact_keys(
        raw["kind"],
        {
            "version",
            "windows_amd64_url",
            "windows_amd64_sha256",
            "filename",
        },
        "kind lock",
    )
    kubectl = _exact_keys(
        raw["kubectl"],
        {
            "version",
            "windows_amd64_url",
            "windows_amd64_sha256",
            "filename",
        },
        "kubectl lock",
    )
    cluster = _exact_keys(
        raw["cluster"],
        {
            "kubernetes_minor",
            "node_image",
            "node_image_source",
            "node_image_id",
            "node_image_dockerfile",
            "node_image_dockerfile_sha256",
            "node_image_entrypoint",
            "node_image_entrypoint_sha256",
            "kind_config_sha256",
            "docker_network_subnet",
            "docker_network_gateway",
        },
        "cluster lock",
    )
    calico = _exact_keys(
        raw["calico"],
        {"version", "manifest_url", "manifest_sha256", "filename"},
        "Calico lock",
    )
    raw_images = raw["preload_images"]
    if not isinstance(raw_images, list) or not raw_images:
        raise ConfigError("preload_images must be a non-empty array")
    images: list[ImageLock] = []
    for index, item in enumerate(raw_images):
        image = _exact_keys(item, {"tag", "source", "digest"}, f"image {index}")
        tag = str(image["tag"])
        source = str(image["source"])
        digest = str(image["digest"])
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ConfigError(f"image {index} has an invalid sha256 digest")
        if source != f"{tag}@{digest}":
            raise ConfigError(
                f"image {index} source must be its tag followed by the declared digest"
            )
        images.append(
            ImageLock(
                tag=tag,
                source=source,
                digest=digest,
            )
        )
    if not any(image.tag.endswith("busybox:1.36.1") for image in images):
        raise ConfigError("The environment lock must preload busybox:1.36.1")
    for label, value in (
        ("kind checksum", kind["windows_amd64_sha256"]),
        ("kubectl checksum", kubectl["windows_amd64_sha256"]),
        ("node image Dockerfile checksum", cluster["node_image_dockerfile_sha256"]),
        ("node image entrypoint checksum", cluster["node_image_entrypoint_sha256"]),
        ("kind config checksum", cluster["kind_config_sha256"]),
        ("Calico checksum", calico["manifest_sha256"]),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise ConfigError(f"{label} must be a lowercase sha256 value")
    if not re.search(r"@sha256:[0-9a-f]{64}\Z", str(cluster["node_image_source"])):
        raise ConfigError("cluster.node_image_source must be digest-pinned")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(cluster["node_image_id"])):
        raise ConfigError("cluster.node_image_id must be a sha256 image id")
    node_image = str(cluster["node_image"])
    if "@" in node_image or not re.fullmatch(r"[a-z0-9._/-]+:[a-zA-Z0-9._-]+", node_image):
        raise ConfigError("cluster.node_image must be a local tagged image")
    try:
        network = ipaddress.ip_network(str(cluster["docker_network_subnet"]), strict=True)
        gateway = ipaddress.ip_address(str(cluster["docker_network_gateway"]))
    except ValueError as exc:
        raise ConfigError("The locked Docker subnet or gateway is invalid") from exc
    if network.version != 4 or gateway not in network or gateway == network.network_address:
        raise ConfigError("The locked Docker gateway must be usable inside its IPv4 subnet")
    pod_network = ipaddress.ip_network("192.168.0.0/16")
    if network.overlaps(pod_network):
        raise ConfigError("The Docker bridge subnet overlaps the locked Pod subnet")
    return EnvironmentLock(
        path=path,
        kind_version=str(kind["version"]),
        kind_url=str(kind["windows_amd64_url"]),
        kind_sha256=str(kind["windows_amd64_sha256"]),
        kind_filename=str(kind["filename"]),
        kubectl_version=str(kubectl["version"]),
        kubectl_url=str(kubectl["windows_amd64_url"]),
        kubectl_sha256=str(kubectl["windows_amd64_sha256"]),
        kubectl_filename=str(kubectl["filename"]),
        kubernetes_minor=str(cluster["kubernetes_minor"]),
        node_image=node_image,
        node_image_source=str(cluster["node_image_source"]),
        node_image_id=str(cluster["node_image_id"]),
        node_image_dockerfile=_project_path(
            cluster["node_image_dockerfile"], "cluster.node_image_dockerfile"
        ),
        node_image_dockerfile_sha256=str(cluster["node_image_dockerfile_sha256"]),
        node_image_entrypoint=_project_path(
            cluster["node_image_entrypoint"], "cluster.node_image_entrypoint"
        ),
        node_image_entrypoint_sha256=str(cluster["node_image_entrypoint_sha256"]),
        kind_config_sha256=str(cluster["kind_config_sha256"]),
        docker_network_subnet=str(cluster["docker_network_subnet"]),
        docker_network_gateway=str(cluster["docker_network_gateway"]),
        calico_version=str(calico["version"]),
        calico_manifest_url=str(calico["manifest_url"]),
        calico_manifest_sha256=str(calico["manifest_sha256"]),
        calico_filename=str(calico["filename"]),
        preload_images=tuple(images),
    )


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path).resolve()
    raw = _read_object(config_path)
    _exact_keys(
        raw,
        {"schema_version", "api", "pipeline", "environment", "prompts"},
        "configuration",
    )
    if raw["schema_version"] != 1:
        raise ConfigError("Unsupported configuration schema_version")

    api = _exact_keys(
        raw["api"],
        {
            "base_url",
            "model",
            "provider_only",
            "allow_fallbacks",
            "temperature",
            "transport_retries",
            "input_usd_per_million",
            "output_usd_per_million",
        },
        "api",
    )
    providers = api["provider_only"]
    if not isinstance(providers, list) or not providers or any(
        not isinstance(item, str) or not item.strip() for item in providers
    ):
        raise ConfigError("api.provider_only must be a non-empty string array")
    base_url = str(api["base_url"]).rstrip("/")
    if base_url != FROZEN_API_BASE:
        raise ConfigError(
            "The live API base is locked to the official OpenRouter HTTPS endpoint"
        )
    model = str(api["model"])
    if model != FROZEN_MODEL:
        raise ConfigError(f"The frozen baseline model must be {FROZEN_MODEL}")
    normalized_providers = tuple(item.strip().lower() for item in providers)
    if normalized_providers != FROZEN_PROVIDERS:
        raise ConfigError("The frozen baseline provider must be Cloudflare only")
    if api["allow_fallbacks"] is not False:
        raise ConfigError("Frozen experiments require allow_fallbacks=false")
    retries = api["transport_retries"]
    if isinstance(retries, bool) or retries not in {0, 1}:
        raise ConfigError("api.transport_retries must be 0 or 1")
    temperature = float(api["temperature"])
    if temperature != 0.0:
        raise ConfigError("The frozen baseline requires temperature=0")
    input_price = _decimal(api["input_usd_per_million"], "input token price")
    output_price = _decimal(api["output_usd_per_million"], "output token price")
    if input_price != FROZEN_INPUT_PRICE or output_price != FROZEN_OUTPUT_PRICE:
        raise ConfigError("The frozen baseline price snapshot was modified")

    pipeline = _exact_keys(
        raw["pipeline"],
        {
            "max_regenerations",
            "runtime_observation_seconds",
            "runtime_poll_seconds",
            "command_timeout_seconds",
            "output_root",
        },
        "pipeline",
    )
    if pipeline["max_regenerations"] != 1:
        raise ConfigError("The baseline is locked to exactly one regeneration")
    for key in (
        "runtime_observation_seconds",
        "runtime_poll_seconds",
        "command_timeout_seconds",
    ):
        if isinstance(pipeline[key], bool) or float(pipeline[key]) <= 0:
            raise ConfigError(f"pipeline.{key} must be above zero")

    environment = _exact_keys(
        raw["environment"], {"lock_file", "kind_config", "cache_root"}, "environment"
    )
    lock_path = _project_path(environment["lock_file"], "environment.lock_file")
    kind_config = _project_path(environment["kind_config"], "environment.kind_config")
    cache_root = _project_path(
        environment["cache_root"], "environment.cache_root", must_exist=False
    )
    prompts = _exact_keys(raw["prompts"], {"generate", "regenerate"}, "prompts")

    return AppConfig(
        path=config_path,
        api=ApiConfig(
            base_url=base_url,
            model=model,
            provider_only=normalized_providers,
            allow_fallbacks=False,
            temperature=temperature,
            transport_retries=int(retries),
            input_usd_per_million=input_price,
            output_usd_per_million=output_price,
        ),
        pipeline=PipelineSettings(
            max_regenerations=1,
            runtime_observation_seconds=int(pipeline["runtime_observation_seconds"]),
            runtime_poll_seconds=float(pipeline["runtime_poll_seconds"]),
            command_timeout_seconds=int(pipeline["command_timeout_seconds"]),
            output_root=_project_path(
                pipeline["output_root"], "pipeline.output_root", must_exist=False
            ),
        ),
        environment=EnvironmentConfig(
            lock=load_environment_lock(lock_path),
            kind_config=kind_config,
            cache_root=cache_root,
        ),
        prompts=PromptConfig(
            generate=_project_path(prompts["generate"], "prompts.generate"),
            regenerate=_project_path(prompts["regenerate"], "prompts.regenerate"),
        ),
    )
