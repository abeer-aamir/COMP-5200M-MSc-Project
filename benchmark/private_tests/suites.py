from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import yaml

from .core import (
    EvaluationError,
    EvaluationInfrastructureError,
    Kubectl,
    KubectlError,
    RequirementFailure,
    ResultCollector,
    all_containers,
    assert_claim,
    assert_exact_role,
    assert_hardened,
    command_text,
    find_mount,
    find_volume,
    int_or_string,
    job_terminal_condition,
    pod_labels,
    pod_spec,
    require,
    role_bound_to_service_account,
    volume_claim_template,
)


EXPECTED_REQUIREMENTS = {
    "pilot-001": {f"R{number:02d}" for number in range(1, 15)},
    "pilot-002": {f"R{number:02d}" for number in range(1, 15)},
    "pilot-003": {f"R{number:02d}" for number in range(1, 12)},
    "pilot-004": {f"R{number:02d}" for number in range(1, 14)},
    "easy-001": {f"R{number:02d}" for number in range(1, 7)},
    "easy-002": {f"R{number:02d}" for number in range(1, 7)},
    "easy-003": {f"R{number:02d}" for number in range(1, 7)},
    "easy-004": {f"R{number:02d}" for number in range(1, 7)},
    "easy-005": {f"R{number:02d}" for number in range(1, 7)},
    "medium-001": {f"R{number:02d}" for number in range(1, 11)},
    "medium-002": {f"R{number:02d}" for number in range(1, 11)},
    "medium-003": {f"R{number:02d}" for number in range(1, 11)},
    "medium-004": {f"R{number:02d}" for number in range(1, 10)},
    "medium-005": {f"R{number:02d}" for number in range(1, 11)},
}

_FRESH_CLUSTER_NAMESPACES = {
    "default",
    "kube-node-lease",
    "kube-public",
    "kube-system",
    "local-path-storage",
}

_DEFAULT_NAMESPACE_BASELINE = {
    "configmaps": {"kube-root-ca.crt"},
    "serviceaccounts": {"default"},
    "services": {"kubernetes"},
}

_DEFAULT_NAMESPACE_EMPTY_KINDS = {
    "cronjobs",
    "daemonsets",
    "deployments",
    "jobs",
    "networkpolicies",
    "persistentvolumeclaims",
    "poddisruptionbudgets",
    "pods",
    "replicasets",
    "replicationcontrollers",
    "resourcequotas",
    "rolebindings",
    "roles",
    "secrets",
    "statefulsets",
}


def _network_policy_rule_has_exact_ports(
    rule: dict[str, Any], expected: set[tuple[str, str]]
) -> bool:
    """Compare the effective port allowance without prescribing peer syntax."""

    ports = rule.get("ports", [])
    if not isinstance(ports, list) or len(ports) != len(expected):
        return False
    actual: set[tuple[str, str]] = set()
    for item in ports:
        if (
            not isinstance(item, dict)
            or "port" not in item
            or item.get("endPort") is not None
        ):
            return False
        actual.add(
            (item.get("protocol", "TCP"), int_or_string(item.get("port")))
        )
    return actual == expected


def _has_bounded_retry_and_nonzero_failure(
    command: str, *, maximum_attempts: int
) -> bool:
    """Recognize common finite BusyBox-shell retry forms without fixing one syntax."""

    limits: list[int] = []
    for pattern in (
        r"\bseq\s+(?:1\s+)?([0-9]+)\b",
        r"-(?:lt|ge)\s+([0-9]+)\b",
    ):
        limits.extend(int(value) for value in re.findall(pattern, command))
    bounded = any(1 <= value <= maximum_attempts for value in limits)
    nonzero_failure = bool(re.search(r"\bexit\s+[1-9][0-9]*\b", command))
    return bounded and nonzero_failure


def _main_container(workload: dict[str, Any]) -> dict[str, Any]:
    containers = pod_spec(workload).get("containers", [])
    require(bool(containers), "workload has no main container")
    return containers[0]


def _cron_spec(cronjob: dict[str, Any]) -> dict[str, Any]:
    return cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]


def _cron_job_spec(cronjob: dict[str, Any]) -> dict[str, Any]:
    return cronjob["spec"]["jobTemplate"]["spec"]


def _service_port(service: dict[str, Any], port: int) -> dict[str, Any]:
    for candidate in service.get("spec", {}).get("ports", []):
        if candidate.get("port") == port:
            return candidate
    raise RequirementFailure(f"Service does not expose port {port}")


def _resolved_target_port(
    service_port: dict[str, Any], workload: dict[str, Any]
) -> str:
    target = service_port.get("targetPort", service_port.get("port"))
    if isinstance(target, str) and not target.isdigit():
        matches = {
            item.get("containerPort")
            for container in pod_spec(workload).get("containers", [])
            for item in container.get("ports", [])
            if item.get("name") == target
        }
        require(
            len(matches) == 1,
            f"named targetPort {target!r} does not resolve unambiguously",
        )
        target = matches.pop()
    return int_or_string(target)


def _container_exposes_configmap_keys(
    container: dict[str, Any], configmap_name: str, keys: set[str]
) -> bool:
    for source in container.get("envFrom", []) or []:
        if not isinstance(source, dict) or source.get("prefix"):
            continue
        if source.get("configMapRef", {}).get("name") == configmap_name:
            return True

    mappings: dict[str, tuple[str | None, str | None]] = {}
    for item in container.get("env", []) or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        ref = item.get("valueFrom", {}).get("configMapKeyRef", {})
        mappings[item["name"]] = (ref.get("name"), ref.get("key"))
    return all(mappings.get(key) == (configmap_name, key) for key in keys)


def _decode_secret_value(encoded: Any, key: str) -> str:
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, TypeError, UnicodeDecodeError, ValueError) as exc:
        raise RequirementFailure(f"Secret key {key} is not valid base64 UTF-8") from exc


def _assert_exact_configmap_data(configmap: dict[str, Any], expected: dict[str, str]) -> None:
    require(configmap.get("data", {}) == expected, "ConfigMap data differs")
    require(not configmap.get("binaryData"), "ConfigMap contains unexpected binaryData keys")


def _container_exposes_secret_key(
    container: dict[str, Any], secret_name: str, key: str, env_name: str
) -> bool:
    explicit = [
        item
        for item in container.get("env", []) or []
        if isinstance(item, dict) and item.get("name") == env_name
    ]
    explicit_match = False
    if len(explicit) == 1:
        ref = explicit[0].get("valueFrom", {}).get("secretKeyRef", {})
        explicit_match = ref.get("name") == secret_name and ref.get("key") == key

    env_from = [
        item
        for item in container.get("envFrom", []) or []
        if isinstance(item, dict)
        and not item.get("prefix")
        and item.get("secretRef", {}).get("name") == secret_name
    ]
    return (explicit_match and not env_from) or (not explicit and len(env_from) == 1)


def _assert_secret_not_in_environment(
    spec: dict[str, Any], secret_name: str
) -> None:
    for container in all_containers(spec):
        label = container.get("name", "<unnamed>")
        for item in container.get("env", []) or []:
            if not isinstance(item, dict):
                continue
            require(
                item.get("valueFrom", {})
                .get("secretKeyRef", {})
                .get("name")
                != secret_name,
                f"Secret {secret_name} is exposed in container {label} through env",
            )
        for item in container.get("envFrom", []) or []:
            if not isinstance(item, dict):
                continue
            require(
                item.get("secretRef", {}).get("name") != secret_name,
                f"Secret {secret_name} is exposed in container {label} through envFrom",
            )


def _startup_probe_window_seconds(probe: dict[str, Any]) -> int:
    values = {
        "initialDelaySeconds": probe.get("initialDelaySeconds", 0),
        "failureThreshold": probe.get("failureThreshold", 3),
        "periodSeconds": probe.get("periodSeconds", 10),
    }
    for name, value in values.items():
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"startupProbe {name} must be a non-negative integer",
        )
    return values["initialDelaySeconds"] + (
        values["failureThreshold"] * values["periodSeconds"]
    )


def _volume_for_mount(spec: dict[str, Any], container: dict[str, Any], path: str) -> dict[str, Any]:
    mount = find_mount(container, path)
    require(mount is not None, f"no volume is mounted at {path}")
    volume = find_volume(spec, mount.get("name", ""))
    require(volume is not None, f"volume {mount.get('name')!r} does not exist")
    return volume


def _policy_selecting(kube: Kubectl, role: str, direction: str) -> dict[str, Any]:
    for policy in kube.list("networkpolicies"):
        selector = policy.get("spec", {}).get("podSelector", {}).get("matchLabels", {})
        policy_types = set(policy.get("spec", {}).get("policyTypes", []))
        if selector.get("role") == role and direction in policy_types:
            return policy
    raise RequirementFailure(f"no {direction} NetworkPolicy selects role={role}")


def _pdb_for_role(kube: Kubectl, role: str) -> dict[str, Any]:
    for pdb in kube.list("poddisruptionbudgets"):
        labels = pdb.get("spec", {}).get("selector", {}).get("matchLabels", {})
        if labels.get("role") == role:
            return pdb
    raise RequirementFailure(f"no PodDisruptionBudget selects role={role}")


def _wait_deployment_ready(kube: Kubectl, name: str, replicas: int, timeout: int = 180) -> None:
    kube.wait_until(
        lambda: kube.get("deployment", name).get("status", {}).get("readyReplicas", 0)
        == replicas,
        f"Deployment {name} to have {replicas} ready replicas",
        timeout,
    )


def _start_sink(kube: Kubectl, name: str = "aipc-eval-sink") -> str:
    kube.delete("pod", name)
    kube.run(
        [
            "run",
            name,
            "-n",
            kube.namespace,
            "--image=busybox:1.36.1",
            "--restart=Never",
            "--labels=role=sink",
            "--command",
            "--",
            "sh",
            "-ec",
            "mkdir -p /tmp/www; printf ok >/tmp/www/health; exec httpd -f -p 8081 -h /tmp/www",
        ]
    )
    state: dict[str, Any] = {}

    def ready() -> bool:
        nonlocal state
        state = kube.get("pod", name)
        statuses = state.get("status", {}).get("containerStatuses", [])
        return bool(state.get("status", {}).get("podIP")) and any(
            item.get("ready") for item in statuses
        )

    kube.wait_until(ready, "evaluator sink pod to become ready", 45, 1)
    return state["status"]["podIP"]


def _assert_pod_label_and_service_account(
    workload: dict[str, Any], role: str, service_account: str | None = None
) -> None:
    require(pod_labels(workload).get("role") == role, f"pod label role={role} missing")
    if service_account:
        actual = pod_spec(workload).get("serviceAccountName", "default")
        require(actual == service_account, f"expected ServiceAccount {service_account}, got {actual}")


def _assert_no_secret_volume(spec: dict[str, Any], secret_name: str) -> None:
    for volume in spec.get("volumes", []):
        require(
            volume.get("secret", {}).get("secretName") != secret_name,
            f"Secret {secret_name} is mounted as a Secret volume",
        )
        for source in volume.get("projected", {}).get("sources", []):
            require(
                source.get("secret", {}).get("name") != secret_name,
                f"Secret {secret_name} is mounted through a projected volume",
            )


def _manifest_resources(document: Any) -> list[dict[str, Any]]:
    """Flatten Kubernetes List documents without consulting live cluster baselines."""

    require(isinstance(document, dict), "candidate contains a non-object YAML document")
    kind = document.get("kind")
    require(isinstance(kind, str) and bool(kind.strip()), "candidate resource has no kind")
    if kind == "List" or kind.endswith("List"):
        items = document.get("items")
        require(isinstance(items, list), f"{kind} resource has no items list")
        flattened: list[dict[str, Any]] = []
        for item in items:
            flattened.extend(_manifest_resources(item))
        return flattened
    return [document]


def _assert_candidate_manifest_scope(
    kube: Kubectl, candidate_path: Path, expected: str
) -> str:
    """Require every candidate-declared object to belong to the task namespace."""

    try:
        with candidate_path.open("r", encoding="utf-8") as handle:
            documents = list(yaml.safe_load_all(handle))
    except OSError as exc:
        raise EvaluationInfrastructureError(
            f"cannot read the deployed candidate manifest: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise RequirementFailure(f"candidate manifest cannot be parsed: {exc}") from exc

    resources = [
        resource
        for document in documents
        if document is not None
        for resource in _manifest_resources(document)
    ]
    require(bool(resources), "candidate manifest declares no Kubernetes resources")

    for resource in resources:
        kind = resource["kind"]
        api_version = resource.get("apiVersion")
        metadata = resource.get("metadata")
        require(isinstance(metadata, dict), f"{kind} resource has no metadata object")
        name = metadata.get("name", "<unnamed>")
        declared_namespace = metadata.get("namespace")
        namespaced = kube.resource_is_namespaced(api_version, kind)
        if not namespaced:
            require(
                api_version == "v1"
                and kind == "Namespace"
                and name == expected
                and declared_namespace is None,
                f"candidate cluster-scoped {api_version}/{kind} {name!r} is not "
                f"the allowed Namespace {expected!r}",
            )
            continue
        require(
            declared_namespace == expected,
            f"candidate {kind} {name!r} declares namespace "
            f"{declared_namespace or '<default>'!r}; expected {expected!r}",
        )

    return f"all {len(resources)} candidate resources are scoped to {expected}"


def _check_namespace(kube: Kubectl, expected: str, candidate_path: Path) -> str:
    namespace = kube.get("namespace", expected, namespace=False)
    require(namespace.get("metadata", {}).get("name") == expected, "namespace is absent")
    require(namespace.get("status", {}).get("phase") == "Active", "namespace is not Active")
    scope = _assert_candidate_manifest_scope(kube, candidate_path, expected)

    namespace_names = {
        item.get("metadata", {}).get("name")
        for item in kube.list("namespaces", namespace=False)
    }
    unexpected_namespaces = namespace_names - (_FRESH_CLUSTER_NAMESPACES | {expected})
    require(
        not unexpected_namespaces,
        f"unexpected namespaces exist: {sorted(unexpected_namespaces)}",
    )
    for kind, allowed_names in _DEFAULT_NAMESPACE_BASELINE.items():
        actual_names = {
            item.get("metadata", {}).get("name")
            for item in kube.list(kind, namespace="default")
        }
        require(
            actual_names <= allowed_names,
            f"unexpected {kind} exist in default: "
            f"{sorted(actual_names - allowed_names)}",
        )
    for kind in sorted(_DEFAULT_NAMESPACE_EMPTY_KINDS):
        items = kube.list(kind, namespace="default")
        require(
            not items,
            f"unexpected {kind} exist in default: "
            f"{sorted(item.get('metadata', {}).get('name') for item in items)}",
        )
    return f"namespace {expected} is Active; {scope}; live isolation is preserved"


def _task1_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        statefulset = kube.get("statefulset", "cache")
        service = kube.get("service", "cache-headless")
        require(statefulset["spec"].get("replicas") == 2, "cache must have 2 replicas")
        _assert_pod_label_and_service_account(statefulset, "cache")
        require(service["spec"].get("clusterIP") == "None", "cache Service is not headless")
        port = _service_port(service, 6379)
        require(port.get("protocol", "TCP") == "TCP", "cache Service port is not TCP")
        claim = volume_claim_template(statefulset, "cache-data")
        assert_claim(claim)
        container = _main_container(statefulset)
        mount = find_mount(container, "/data")
        require(mount is not None and mount.get("name") == "cache-data", "cache-data is not mounted at /data")
        return "cache StatefulSet, headless Service, and per-pod storage are correct"

    def r03() -> str:
        deployment = kube.get("deployment", "api")
        require(deployment["spec"].get("replicas") == 3, "api must have 3 replicas")
        _assert_pod_label_and_service_account(deployment, "api", "api-sa")
        container = _main_container(deployment)
        ports = {item.get("containerPort") for item in container.get("ports", [])}
        require(8080 in ports, "api container does not expose port 8080")
        probe = container.get("readinessProbe", {})
        require("exec" in probe, "api readiness probe must be exec")
        probe_text = " ".join(probe.get("exec", {}).get("command", []))
        for value in ("cache-0.cache-headless", "6379", "/health"):
            require(value in probe_text, f"api readiness probe does not reference {value}")
        kube.rollout("statefulset", "cache")
        kube.rollout("deployment", "api")
        return "api has 3 ready replicas and an in-pod cache readiness check"

    def r04() -> str:
        config = kube.get("configmap", "api-config")
        secret = kube.get("secret", "api-credentials")
        deployment = kube.get("deployment", "api")
        expected = {
            "APP_MODE": "production",
            "CACHE_HOST": "cache-headless",
            "CACHE_PORT": "6379",
        }
        require(config.get("data") == expected, f"api-config data differs: {config.get('data')}")
        encoded = secret.get("data", {}).get("API_KEY", "")
        require(
            _decode_secret_value(encoded, "API_KEY") == "pilot-token",
            "API_KEY placeholder differs",
        )
        spec = pod_spec(deployment)
        container = _main_container(deployment)
        require(
            _container_exposes_configmap_keys(container, "api-config", set(expected)),
            "api-config is not exposed as the requested environment variables",
        )
        volume = _volume_for_mount(spec, container, "/etc/api/secrets")
        projected = volume.get("projected", {})
        require(projected.get("defaultMode") == 0o440, "projected Secret mode is not 0440")
        names = {
            source.get("secret", {}).get("name") for source in projected.get("sources", [])
        }
        require("api-credentials" in names, "api-credentials is not projected")
        _assert_secret_not_in_environment(spec, "api-credentials")
        return "ConfigMap environment and projected Secret are wired correctly"

    def r05() -> str:
        deployment = kube.get("deployment", "api")
        init = pod_spec(deployment).get("initContainers", [])
        require(bool(init), "api has no init container")
        text = " ".join(command_text(item) for item in init)
        for value in ("cache-0.cache-headless", "cache-1.cache-headless", "6379"):
            require(value in text, f"api init logic does not reference {value}")
        require("kubectl" not in text, "api init container must not query the Kubernetes API")
        return "api init container waits for both stable cache endpoints"

    def r06() -> str:
        service = kube.get("service", "api-svc")
        require(service["spec"].get("type", "ClusterIP") == "ClusterIP", "api-svc is not ClusterIP")
        port = _service_port(service, 80)
        require(
            _resolved_target_port(port, kube.get("deployment", "api")) == "8080",
            "api-svc targetPort does not resolve to 8080",
        )
        probe = kube.run_probe_pod(
            "aipc-eval-api-service",
            "wget -q -T 5 -O /dev/null http://api-svc:80/health",
        )
        return f"api-svc routes to the running API; {probe}"

    def r07() -> str:
        deployment = kube.get("deployment", "api")
        require(pod_spec(deployment).get("serviceAccountName") == "api-sa", "api does not use api-sa")
        role = role_bound_to_service_account(kube, "api-sa")
        assert_exact_role(role, {"get", "list", "watch"}, {"pods", "configmaps"})
        allowed = all(
            kube.auth_can_i("api-sa", verb, resource)
            for verb in ("get", "list", "watch")
            for resource in ("pods", "configmaps")
        )
        denied = all(
            not kube.auth_can_i("api-sa", verb, resource)
            for verb, resource in (
                ("create", "pods"),
                ("delete", "configmaps"),
                ("get", "secrets"),
                ("get", "deployments"),
            )
        )
        require(allowed and denied, "live RBAC authorization differs from the least-privilege role")
        for pod in kube.list("pods"):
            if pod.get("spec", {}).get("serviceAccountName") == "api-sa":
                require(pod.get("metadata", {}).get("labels", {}).get("role") == "api", "api-sa is used by a non-api pod")
        return "api-sa live permissions are exactly the requested read-only set"

    def r08() -> str:
        policy = _policy_selecting(kube, "api", "Egress")
        require(policy.get("spec", {}).get("egress"), "api Egress policy has no allow rules")
        allowed = kube.run_probe_pod(
            "aipc-eval-api-allowed",
            "nslookup kubernetes.default.svc.cluster.local >/dev/null && wget -q -T 5 -O /dev/null http://cache-0.cache-headless:6379/health",
            labels={"role": "api"},
        )
        sink_ip = _start_sink(kube)
        try:
            denied = kube.run_probe_pod(
                "aipc-eval-api-denied",
                f"wget -q -T 5 -O /dev/null http://{sink_ip}:8081/health",
                labels={"role": "api"},
                expect_success=False,
            )
        finally:
            kube.delete("pod", "aipc-eval-sink")
        return f"DNS/cache egress allowed and unrelated egress denied; {allowed}; {denied}"

    def r09() -> str:
        policy = _policy_selecting(kube, "cache", "Ingress")
        require(policy.get("spec", {}).get("ingress"), "cache Ingress policy has no allow rules")
        denied = kube.run_probe_pod(
            "aipc-eval-cache-denied",
            "wget -q -T 5 -O /dev/null http://cache-0.cache-headless:6379/health",
            labels={"role": "untrusted"},
            expect_success=False,
        )
        reconciler = kube.run_probe_pod(
            "aipc-eval-cache-reconciler",
            "wget -q -T 5 -O /dev/null http://cache-0.cache-headless:6379/health",
            labels={"role": "reconciler"},
        )
        return f"cache rejects untrusted ingress and permits reconciler ingress; {denied}; {reconciler}"

    def r10() -> str:
        assert_hardened(kube.get("deployment", "api"), "api")
        assert_hardened(kube.get("statefulset", "cache"), "cache")
        return "api and cache main/init containers have the requested security settings"

    def r11() -> str:
        cronjob = kube.get("cronjob", "cache-reconciler")
        require(cronjob["spec"].get("schedule") == "*/5 * * * *", "cache-reconciler schedule differs")
        job_spec = _cron_job_spec(cronjob)
        require(job_spec.get("backoffLimit") == 0, "cache-reconciler backoffLimit must be 0")
        deadline = job_spec.get("activeDeadlineSeconds")
        require(isinstance(deadline, int) and deadline <= 60, "cache-reconciler deadline must be <=60s")
        template = job_spec["template"]
        require(template.get("metadata", {}).get("labels", {}).get("role") == "reconciler", "reconciler pod label is missing")
        text = " ".join(command_text(item) for item in template["spec"].get("containers", []))
        for value in ("cache-0.cache-headless", "cache-1.cache-headless", "6379"):
            require(value in text, f"reconciler does not check {value}")
        kube.create_job_from_cronjob("cache-reconciler", "cache-healthy", expect_success=True)
        kube.scale("statefulset", "cache", 1)
        try:
            kube.wait_until(
                lambda: kube.get("statefulset", "cache").get("status", {}).get("readyReplicas", 0) == 1,
                "cache to scale down to one ready replica",
                120,
            )
            kube.create_job_from_cronjob("cache-reconciler", "cache-unhealthy", expect_success=False)
        finally:
            kube.scale("statefulset", "cache", 2)
            kube.rollout("statefulset", "cache")
        return "reconciler succeeds with two cache pods and fails with one"

    def r12() -> str:
        api_pdb = _pdb_for_role(kube, "api")
        cache_pdb = _pdb_for_role(kube, "cache")
        require(int_or_string(api_pdb["spec"].get("minAvailable")) == "2", "api PDB minAvailable differs")
        require(int_or_string(cache_pdb["spec"].get("minAvailable")) == "1", "cache PDB minAvailable differs")
        return "both disruption budgets have the requested availability floors"

    def r13() -> str:
        deployment = kube.get("deployment", "api")
        actual = deployment["spec"]["template"]["metadata"].get("annotations", {}).get("checksum/api-config")
        source = b"APP_MODE=production\nCACHE_HOST=cache-headless\nCACHE_PORT=6379\n"
        expected = hashlib.sha256(source).hexdigest()
        require(actual == expected, f"checksum/api-config must be {expected}, got {actual!r}")
        return f"pod-template checksum matches api-config ({expected})"

    def r14() -> str:
        quota = kube.get("resourcequota", next(
            (item["metadata"]["name"] for item in kube.list("resourcequotas")), ""
        ))
        hard = quota.get("status", {}).get("hard", quota.get("spec", {}).get("hard", {}))
        require(hard.get("pods") == "10", f"pod quota must be 10, got {hard.get('pods')}")
        require(hard.get("requests.storage") == "5Gi", "storage quota must be 5Gi")
        used_pods = int(quota.get("status", {}).get("used", {}).get("pods", "0"))
        require(used_pods <= 10, "current workloads exceed the pod quota")
        return f"quota is correct and current pod use is {used_pods}/10"

    return [
        (
            "R01",
            "isolated order-system namespace",
            lambda: _check_namespace(
                kube,
                "order-system",
                _require_candidate_path(candidate_path),
            ),
        ),
        ("R02", "cache StatefulSet, Service, and storage", r02),
        ("R03", "API replicas, port, and dependency readiness", r03),
        ("R04", "API configuration and Secret projection", r04),
        ("R05", "API init dependency gate", r05),
        ("R06", "API ClusterIP Service connectivity", r06),
        ("R07", "least-privilege API RBAC", r07),
        ("R08", "API egress isolation", r08),
        ("R09", "cache ingress isolation", r09),
        ("R10", "API and cache security contexts", r10),
        ("R11", "cache reconciliation success and failure", r11),
        ("R12", "API and cache disruption budgets", r12),
        ("R13", "configuration rollout checksum", r13),
        ("R14", "namespace quota and live fit", r14),
    ]


def _task2_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        config = kube.get("configmap", "pipeline-config")
        require(config.get("data") == {"WORKER_COUNT": "3", "POLL_INTERVAL": "5"}, "pipeline-config data differs")
        for kind, name in (("deployment", "controller"), ("statefulset", "worker")):
            workload = kube.get(kind, name)
            spec = pod_spec(workload)
            volume = _volume_for_mount(spec, _main_container(workload), "/etc/pipeline")
            require(volume.get("configMap", {}).get("name") == "pipeline-config", f"{name} does not mount pipeline-config")
        return "pipeline-config data is mounted in controller and workers"

    def r03() -> str:
        secret = kube.get("secret", "pipeline-secret")
        encoded = secret.get("data", {}).get("API_TOKEN", "")
        require(
            _decode_secret_value(encoded, "API_TOKEN") == "pilot-token",
            "API_TOKEN placeholder differs",
        )
        controller = kube.get("deployment", "controller")
        controller_spec = pod_spec(controller)
        controller_container = _main_container(controller)
        require(
            _container_exposes_secret_key(
                controller_container,
                "pipeline-secret",
                "API_TOKEN",
                "API_TOKEN",
            ),
            "controller API_TOKEN is not sourced exactly once from pipeline-secret",
        )
        _assert_no_secret_volume(controller_spec, "pipeline-secret")
        for container in all_containers(controller_spec):
            if container is controller_container:
                continue
            _assert_secret_not_in_environment(
                {"containers": [container]}, "pipeline-secret"
            )
        for workload in (
            kube.get("statefulset", "worker"),
            kube.get("cronjob", "marker-job"),
        ):
            spec = (
                _cron_spec(workload)
                if workload.get("kind") == "CronJob"
                else pod_spec(workload)
            )
            _assert_no_secret_volume(spec, "pipeline-secret")
            _assert_secret_not_in_environment(spec, "pipeline-secret")
        return "API_TOKEN is controller-only environment data and is never mounted"

    def r04() -> str:
        service = kube.get("service", "worker-svc")
        require(service["spec"].get("clusterIP") == "None", "worker-svc is not headless")
        require(service["spec"].get("selector", {}).get("role") == "worker", "worker-svc does not select workers")
        port = _service_port(service, 8080)
        require(port.get("protocol", "TCP") == "TCP", "worker-svc port is not TCP")
        require(
            _resolved_target_port(port, kube.get("statefulset", "worker"))
            == "8080",
            "worker-svc targetPort does not resolve to 8080",
        )
        return "headless worker Service selects worker pods on TCP 8080"

    def r05() -> str:
        statefulset = kube.get("statefulset", "worker")
        require(statefulset["spec"].get("replicas") == 3, "worker must have 3 replicas")
        require(statefulset["spec"].get("serviceName") == "worker-svc", "worker serviceName differs")
        _assert_pod_label_and_service_account(statefulset, "worker")
        claim = volume_claim_template(statefulset, "worker-data")
        assert_claim(claim)
        mount = find_mount(_main_container(statefulset), "/data")
        require(mount is not None and mount.get("name") == "worker-data", "worker-data is not mounted at /data")
        return "worker StatefulSet identity, replicas, and per-pod storage are correct"

    def r06() -> str:
        statefulset = kube.get("statefulset", "worker")
        init = pod_spec(statefulset).get("initContainers", [])
        require(bool(init), "worker has no init container")
        require("/data/ready" in " ".join(command_text(item) for item in init), "worker init does not wait for /data/ready")
        cron = kube.get("cronjob", "marker-job")
        spec = _cron_spec(cron)
        container = spec.get("containers", [None])[0]
        require(container is not None, "marker-job has no container")
        expected = {
            "/worker-data/0": "worker-data-worker-0",
            "/worker-data/1": "worker-data-worker-1",
            "/worker-data/2": "worker-data-worker-2",
        }
        for path, claim in expected.items():
            volume = _volume_for_mount(spec, container, path)
            require(volume.get("persistentVolumeClaim", {}).get("claimName") == claim, f"{path} does not mount {claim}")
        text = command_text(container)
        for path in expected:
            require(f"{path}/ready" in text, f"marker command does not create {path}/ready")
        return "worker init gate and marker-job PVC path convention agree"

    def r07() -> str:
        cron = kube.get("cronjob", "marker-job")
        require(cron["spec"].get("schedule") == "*/2 * * * *", "marker-job schedule differs")
        deadline = _cron_job_spec(cron).get("activeDeadlineSeconds")
        require(isinstance(deadline, int) and deadline <= 60, "marker-job deadline must be <=60s")
        kube.create_job_from_cronjob("marker-job", "markers", expect_success=True, timeout_seconds=65)
        kube.rollout("statefulset", "worker")
        pods = sorted(kube.list("pods", "role=worker"), key=lambda item: item["metadata"]["name"])
        require(len(pods) == 3, "three worker pods were not created")
        for pod in pods:
            kube.exec(pod["metadata"]["name"], "test -f /data/ready")
        return "manual marker Job completed within the deadline and all workers see their marker"

    def r08() -> str:
        deployment = kube.get("deployment", "controller")
        require(deployment["spec"].get("replicas") == 1, "controller must have exactly 1 replica")
        _assert_pod_label_and_service_account(deployment, "controller", "controller-sa")
        probe = _main_container(deployment).get("readinessProbe", {})
        require("exec" in probe, "controller readiness must be an exec probe")
        text = " ".join(probe.get("exec", {}).get("command", []))
        for value in (
            "worker-0.worker-svc",
            "worker-1.worker-svc",
            "worker-2.worker-svc",
            "8080",
            "/health",
        ):
            require(value in text, f"controller readiness does not reference {value}")
        _wait_deployment_ready(kube, "controller", 1)
        return "controller is Ready and its in-pod probe counts three stable worker endpoints"

    def r09() -> str:
        deployment = kube.get("deployment", "controller")
        require(pod_spec(deployment).get("serviceAccountName") == "controller-sa", "controller does not use controller-sa")
        role = role_bound_to_service_account(kube, "controller-sa")
        assert_exact_role(role, {"get", "list"}, {"pods", "services"})
        allowed = all(
            kube.auth_can_i("controller-sa", verb, resource)
            for verb in ("get", "list")
            for resource in ("pods", "services")
        )
        denied = all(
            not kube.auth_can_i("controller-sa", verb, resource)
            for verb, resource in (
                ("watch", "pods"),
                ("create", "services"),
                ("get", "secrets"),
                ("get", "deployments"),
            )
        )
        require(allowed and denied, "controller-sa live authorization differs")
        return "controller-sa has exactly get/list on Pods and Services"

    def r10() -> str:
        assert_hardened(kube.get("deployment", "controller"), "controller")
        assert_hardened(kube.get("statefulset", "worker"), "worker")
        return "controller and worker main/init containers are hardened"

    def r11() -> str:
        policy = kube.get("networkpolicy", "restrict-worker")
        selector = policy.get("spec", {}).get("podSelector", {}).get("matchLabels", {})
        require(selector.get("role") == "worker", "restrict-worker does not select worker pods")
        require("Ingress" in set(policy.get("spec", {}).get("policyTypes", [])), "restrict-worker does not isolate ingress")
        allowed = kube.run_probe_pod(
            "aipc-eval-worker-allowed",
            "wget -q -T 5 -O /dev/null http://worker-0.worker-svc:8080/health",
            labels={"role": "controller"},
        )
        denied = kube.run_probe_pod(
            "aipc-eval-worker-denied",
            "wget -q -T 5 -O /dev/null http://worker-0.worker-svc:8080/health",
            labels={"role": "untrusted"},
            expect_success=False,
        )
        return f"controller ingress succeeds and untrusted ingress fails; {allowed}; {denied}"

    def r12() -> str:
        policy = kube.get("networkpolicy", "restrict-egress")
        selector = policy.get("spec", {}).get("podSelector", {}).get("matchLabels", {})
        require(selector.get("role") == "controller", "restrict-egress does not select controller")
        require("Egress" in set(policy.get("spec", {}).get("policyTypes", [])), "restrict-egress does not isolate egress")
        allowed = kube.run_probe_pod(
            "aipc-eval-controller-egress",
            "nslookup kubernetes.default.svc.cluster.local >/dev/null && wget -q -T 5 -O /dev/null http://worker-0.worker-svc:8080/health",
            labels={"role": "controller"},
        )
        sink_ip = _start_sink(kube)
        try:
            denied = kube.run_probe_pod(
                "aipc-eval-controller-denied",
                f"wget -q -T 5 -O /dev/null http://{sink_ip}:8081/health",
                labels={"role": "controller"},
                expect_success=False,
            )
        finally:
            kube.delete("pod", "aipc-eval-sink")
        return f"worker/DNS egress succeeds and unrelated egress fails; {allowed}; {denied}"

    def r13() -> str:
        pdb = kube.get("poddisruptionbudget", "worker-pdb")
        require(pdb.get("spec", {}).get("selector", {}).get("matchLabels", {}).get("role") == "worker", "worker-pdb does not select workers")
        require(int_or_string(pdb["spec"].get("minAvailable")) == "2", "worker-pdb minAvailable differs")
        return "worker-pdb keeps at least two worker pods available"

    def r14() -> str:
        kube.scale("statefulset", "worker", 1)
        try:
            kube.wait_until(
                lambda: kube.get("statefulset", "worker").get("status", {}).get("readyReplicas", 0) == 1,
                "worker StatefulSet to have one ready replica",
                150,
            )
            kube.wait_until(
                lambda: kube.get("deployment", "controller").get("status", {}).get("readyReplicas", 0) == 0,
                "controller to become NotReady with one worker",
                120,
            )
        finally:
            kube.scale("statefulset", "worker", 3)
            kube.rollout("statefulset", "worker")
        _wait_deployment_ready(kube, "controller", 1)
        return "controller became NotReady at one worker and recovered after restoring three"

    return [
        (
            "R01",
            "isolated pipeline-ns namespace",
            lambda: _check_namespace(
                kube,
                "pipeline-ns",
                _require_candidate_path(candidate_path),
            ),
        ),
        ("R02", "shared pipeline ConfigMap volume", r02),
        ("R03", "controller-only Secret environment", r03),
        ("R04", "headless worker Service", r04),
        ("R05", "worker StatefulSet and storage", r05),
        ("R06", "marker and worker init path agreement", r06),
        ("R07", "marker Job completion and persistent markers", r07),
        ("R08", "controller counted-worker readiness", r08),
        ("R09", "least-privilege controller RBAC", r09),
        ("R10", "controller and worker security contexts", r10),
        ("R11", "worker ingress isolation", r11),
        ("R12", "controller egress isolation", r12),
        ("R13", "worker disruption budget", r13),
        ("R14", "dynamic controller readiness", r14),
    ]


def _task3_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        deployment = kube.get("deployment", "status-web")
        require(
            not pod_spec(deployment).get("initContainers"),
            "status-web must not have init containers",
        )
        require(
            len(pod_spec(deployment).get("containers", [])) == 1,
            "status-web must have exactly one application container",
        )
        deployment_names = {
            item.get("metadata", {}).get("name")
            for item in kube.list("deployments")
        }
        require(
            deployment_names == {"status-web"},
            f"unexpected Deployments exist: {sorted(deployment_names)}",
        )
        service_names = {
            item.get("metadata", {}).get("name") for item in kube.list("services")
        }
        require(
            service_names == {"status-web-svc"},
            f"unexpected Services exist: {sorted(service_names)}",
        )
        replica_sets = kube.list("replicasets")
        allowed_replica_sets = {
            item.get("metadata", {}).get("name")
            for item in replica_sets
            if any(
                owner.get("kind") == "Deployment"
                and owner.get("name") == "status-web"
                for owner in item.get("metadata", {}).get("ownerReferences", [])
            )
        }
        require(
            len(allowed_replica_sets) == len(replica_sets),
            "a ReplicaSet is not owned by status-web",
        )
        for pod in kube.list("pods"):
            owners = pod.get("metadata", {}).get("ownerReferences", [])
            require(
                any(
                    owner.get("kind") == "ReplicaSet"
                    and owner.get("name") in allowed_replica_sets
                    for owner in owners
                ),
                f"pod {pod.get('metadata', {}).get('name')} is not owned by status-web",
            )
            require(
                not pod.get("spec", {}).get("initContainers"),
                "an application pod has an init container",
            )
        forbidden = {
            "StatefulSets": kube.list("statefulsets"),
            "DaemonSets": kube.list("daemonsets"),
            "ReplicationControllers": kube.list("replicationcontrollers"),
            "persistent volume claims": kube.list("persistentvolumeclaims"),
            "Jobs": kube.list("jobs"),
            "CronJobs": kube.list("cronjobs"),
        }
        present = [label for label, items in forbidden.items() if items]
        require(not present, f"forbidden workload/storage resources exist: {present}")
        headless = [
            item.get("metadata", {}).get("name", "")
            for item in kube.list("services")
            if item.get("spec", {}).get("clusterIP") == "None"
        ]
        require(not headless, f"headless Services are forbidden: {headless}")
        return "the application has no dependency-gating or stateful topology"

    def r03() -> str:
        config = kube.get("configmap", "site-content")
        expected = {
            "index.html": "AIPyCraft pilot service\n",
            "health": "ok\n",
        }
        require(config.get("data") == expected, f"site-content data differs: {config.get('data')}")
        deployment = kube.get("deployment", "status-web")
        spec = pod_spec(deployment)
        container = _main_container(deployment)
        mount = find_mount(container, "/www")
        require(mount is not None, "site-content is not mounted at /www")
        volume = find_volume(spec, mount.get("name", ""))
        require(volume is not None, f"volume {mount.get('name')!r} does not exist")
        require(
            volume.get("configMap", {}).get("name") == "site-content",
            "/www is not backed by site-content",
        )
        return "exact site content is mounted read-only at /www"

    def r04() -> str:
        secret = kube.get("secret", "site-secret")
        encoded = secret.get("data", {}).get("RELEASE_CHANNEL", "")
        require(
            _decode_secret_value(encoded, "RELEASE_CHANNEL") == "pilot",
            "RELEASE_CHANNEL placeholder differs",
        )
        deployment = kube.get("deployment", "status-web")
        spec = pod_spec(deployment)
        container = _main_container(deployment)
        require(
            _container_exposes_secret_key(
                container,
                "site-secret",
                "RELEASE_CHANNEL",
                "RELEASE_CHANNEL",
            ),
            "RELEASE_CHANNEL is not sourced exactly once from site-secret",
        )
        _assert_no_secret_volume(spec, "site-secret")
        return "site-secret is exposed only through the requested environment variable"

    def r05() -> str:
        deployment = kube.get("deployment", "status-web")
        require(deployment.get("spec", {}).get("replicas") == 2, "status-web must have 2 replicas")
        labels = pod_labels(deployment)
        require(labels.get("app") == "status-web", "pod label app=status-web is missing")
        _assert_pod_label_and_service_account(deployment, "web", "status-web-sa")
        selector = deployment.get("spec", {}).get("selector", {}).get("matchLabels", {})
        require(selector.get("app") == "status-web", "Deployment selector does not match app=status-web")
        container = _main_container(deployment)
        require(container.get("image") == "busybox:1.36.1", "web image must be busybox:1.36.1")
        ports = {item.get("containerPort") for item in container.get("ports", [])}
        require(8080 in ports, "web container does not expose port 8080")
        text = command_text(container)
        for value in ("httpd", "-f", "8080", "/www"):
            require(value in text, f"web command does not reference {value}")
        strategy = deployment.get("spec", {}).get("strategy", {})
        require(strategy.get("type", "RollingUpdate") == "RollingUpdate", "status-web must use RollingUpdate")
        rolling = strategy.get("rollingUpdate", {})
        require(int_or_string(rolling.get("maxUnavailable")) == "0", "maxUnavailable must be 0")
        require(int_or_string(rolling.get("maxSurge")) == "1", "maxSurge must be 1")
        kube.rollout("deployment", "status-web", timeout_seconds=120)
        return "two status-web replicas serve /www with the requested rollout strategy"

    def r06() -> str:
        container = _main_container(kube.get("deployment", "status-web"))
        named_ports = {
            item.get("name"): item.get("containerPort")
            for item in container.get("ports", [])
            if item.get("name")
        }

        def assert_http_probe(name: str) -> dict[str, Any]:
            probe = container.get(name, {})
            http_get = probe.get("httpGet", {})
            require(http_get.get("path") == "/health", f"{name} path must be /health")
            port = http_get.get("port")
            resolved = named_ports.get(port, port)
            require(int_or_string(resolved) == "8080", f"{name} must target port 8080")
            return probe

        startup = assert_http_probe("startupProbe")
        readiness = assert_http_probe("readinessProbe")
        liveness = assert_http_probe("livenessProbe")
        startup_window = _startup_probe_window_seconds(startup)
        require(startup_window >= 30, "startup probe allows less than 30 seconds")
        for name, probe in (("readinessProbe", readiness), ("livenessProbe", liveness)):
            require(probe.get("periodSeconds", 10) <= 10, f"{name} runs less often than every 10 seconds")
        return "startup, readiness, and liveness HTTP probes use /health on port 8080"

    def r07() -> str:
        service = kube.get("service", "status-web-svc")
        require(service.get("spec", {}).get("type", "ClusterIP") == "ClusterIP", "status-web-svc is not ClusterIP")
        require(service.get("spec", {}).get("clusterIP") not in {None, "", "None"}, "status-web-svc has no ClusterIP")
        require(service.get("spec", {}).get("selector", {}).get("app") == "status-web", "Service does not select app=status-web")
        port = _service_port(service, 80)
        require(port.get("protocol", "TCP") == "TCP", "Service port is not TCP")
        require(
            _resolved_target_port(port, kube.get("deployment", "status-web"))
            == "8080",
            "Service targetPort does not resolve to 8080",
        )
        cluster_ip = service["spec"]["clusterIP"]
        probe = kube.run_probe_pod(
            "aipc-eval-status-web",
            "set -eu; "
            f"test \"$(wget -q -T 5 -O - http://{cluster_ip}:80/)\" = \"AIPyCraft pilot service\"; "
            f"test \"$(wget -q -T 5 -O - http://{cluster_ip}:80/health)\" = ok",
        )
        return f"ClusterIP Service returns the exact index and health content; {probe}"

    def r08() -> str:
        deployment = kube.get("deployment", "status-web")
        require(
            pod_spec(deployment).get("serviceAccountName") == "status-web-sa",
            "status-web does not use status-web-sa",
        )
        role = role_bound_to_service_account(kube, "status-web-sa")
        assert_exact_role(role, {"get", "list"}, {"configmaps"})
        allowed = all(
            kube.auth_can_i("status-web-sa", verb, "configmaps")
            for verb in ("get", "list")
        )
        denied = all(
            not kube.auth_can_i("status-web-sa", verb, resource)
            for verb, resource in (
                ("watch", "configmaps"),
                ("create", "configmaps"),
                ("get", "secrets"),
                ("get", "pods"),
                ("get", "deployments"),
            )
        )
        require(allowed and denied, "status-web-sa live authorization differs")
        for pod in kube.list("pods"):
            if pod.get("spec", {}).get("serviceAccountName") == "status-web-sa":
                require(
                    pod.get("metadata", {}).get("labels", {}).get("app") == "status-web",
                    "status-web-sa is used by a non-status-web pod",
                )
        return "status-web-sa has exactly get/list access to ConfigMaps"

    def r09() -> str:
        deployment = kube.get("deployment", "status-web")
        assert_hardened(deployment, "status-web")
        for container in all_containers(pod_spec(deployment)):
            require(
                container.get("securityContext", {}).get("allowPrivilegeEscalation") is False,
                "status-web must disable privilege escalation",
            )
        return "the web container is non-root, read-only, and capability-free"

    def r10() -> str:
        pdb = kube.get("poddisruptionbudget", "status-web-pdb")
        selector = pdb.get("spec", {}).get("selector", {}).get("matchLabels", {})
        require(selector.get("role") == "web", "status-web-pdb does not select role=web")
        require(int_or_string(pdb.get("spec", {}).get("minAvailable")) == "1", "status-web-pdb minAvailable differs")
        return "status-web-pdb keeps at least one web pod available"

    def r11() -> str:
        quota = kube.get("resourcequota", "status-page-quota")
        hard = quota.get("status", {}).get("hard", quota.get("spec", {}).get("hard", {}))
        require(hard.get("pods") == "5", f"pod quota must be 5, got {hard.get('pods')}")
        deployment = kube.get("deployment", "status-web")
        replicas = int(deployment.get("spec", {}).get("replicas", 1))
        surge = int_or_string(
            deployment.get("spec", {})
            .get("strategy", {})
            .get("rollingUpdate", {})
            .get("maxSurge", 1)
        )
        require(surge.isdigit(), "maxSurge must be an integer for quota-fit analysis")
        peak_with_probe = replicas + int(surge) + 1
        require(
            peak_with_probe <= 5,
            f"rollout plus evaluator probe needs {peak_with_probe} pods, exceeding quota 5",
        )
        used_pods = int(quota.get("status", {}).get("used", {}).get("pods", "0"))
        return (
            f"pod quota is 5; computed rollout-plus-probe peak is "
            f"{peak_with_probe}, current use is {used_pods}/5"
        )

    return [
        (
            "R01",
            "isolated status-page namespace",
            lambda: _check_namespace(
                kube,
                "status-page",
                _require_candidate_path(candidate_path),
            ),
        ),
        ("R02", "simple dependency-free topology", r02),
        ("R03", "exact read-only site content", r03),
        ("R04", "environment-only release Secret", r04),
        ("R05", "web Deployment identity and rollout", r05),
        ("R06", "HTTP startup, readiness, and liveness probes", r06),
        ("R07", "live ClusterIP Service responses", r07),
        ("R08", "least-privilege web RBAC", r08),
        ("R09", "web-container security context", r09),
        ("R10", "web disruption budget", r10),
        ("R11", "namespace pod quota and live fit", r11),
    ]


def _task4_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def named_main(workload_name: str, container_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        workload = kube.get("deployment", workload_name)
        containers = pod_spec(workload).get("containers", [])
        require(len(containers) == 1, f"{workload_name} must have exactly one main container")
        require(containers[0].get("name") == container_name, f"{workload_name} container must be named {container_name}")
        return workload, containers[0]

    def assert_configmap_mount(
        deployment: dict[str, Any], container: dict[str, Any], name: str, path: str
    ) -> None:
        mount = find_mount(container, path)
        require(mount is not None, f"{name} is not mounted at {path}")
        require(mount.get("readOnly") is True, f"{path} mount must be read-only")
        volume = find_volume(pod_spec(deployment), mount.get("name", ""))
        require(volume is not None, f"volume {mount.get('name')!r} does not exist")
        require(volume.get("configMap", {}).get("name") == name, f"{path} is not backed by {name}")

    def assert_http_probes(container: dict[str, Any], path: str, port: int) -> None:
        named_ports = {
            item.get("name"): item.get("containerPort")
            for item in container.get("ports", [])
            if item.get("name")
        }
        probes: dict[str, dict[str, Any]] = {}
        for name in ("startupProbe", "readinessProbe", "livenessProbe"):
            probe = container.get(name, {})
            http_get = probe.get("httpGet", {})
            require(http_get.get("path") == path, f"{name} path must be {path}")
            actual_port = named_ports.get(http_get.get("port"), http_get.get("port"))
            require(int_or_string(actual_port) == str(port), f"{name} must target port {port}")
            probes[name] = probe
        require(_startup_probe_window_seconds(probes["startupProbe"]) >= 30, "startup probe allows less than 30 seconds")
        for name in ("readinessProbe", "livenessProbe"):
            require(probes[name].get("periodSeconds", 10) <= 10, f"{name} runs less often than every 10 seconds")

    def policy(name: str) -> dict[str, Any]:
        return kube.get("networkpolicy", name)

    def selector_labels(item: dict[str, Any]) -> dict[str, str]:
        return item.get("spec", {}).get("podSelector", {}).get("matchLabels", {})

    def has_role_expression(item: dict[str, Any], values: set[str]) -> bool:
        selector = item.get("spec", {}).get("podSelector", {})
        expressions = selector.get("matchExpressions", [])
        return (
            not selector.get("matchLabels")
            and len(expressions) == 1
            and expressions[0].get("key") == "role"
            and expressions[0].get("operator") == "In"
            and set(expressions[0].get("values", [])) == values
        )

    def r02() -> str:
        deployments = {item.get("metadata", {}).get("name") for item in kube.list("deployments")}
        services = {item.get("metadata", {}).get("name") for item in kube.list("services")}
        require(deployments == {"backend", "gateway"}, f"unexpected Deployments: {sorted(deployments)}")
        require(services == {"backend-svc", "gateway-svc"}, f"unexpected Services: {sorted(services)}")
        for name, init_count in (("backend", 0), ("gateway", 1)):
            deployment = kube.get("deployment", name)
            require(len(pod_spec(deployment).get("containers", [])) == 1, f"{name} must have one main container")
            require(len(pod_spec(deployment).get("initContainers", [])) == init_count, f"{name} init-container count differs")
        replica_sets = kube.list("replicasets")
        owned_sets = {
            item.get("metadata", {}).get("name")
            for item in replica_sets
            if any(owner.get("kind") == "Deployment" and owner.get("name") in deployments for owner in item.get("metadata", {}).get("ownerReferences", []))
        }
        require(len(owned_sets) == len(replica_sets), "a ReplicaSet is not owned by backend or gateway")
        for pod in kube.list("pods"):
            owners = pod.get("metadata", {}).get("ownerReferences", [])
            require(any(owner.get("kind") == "ReplicaSet" and owner.get("name") in owned_sets for owner in owners), f"pod {pod.get('metadata', {}).get('name')} is not owned by an application Deployment")
        forbidden = {
            "StatefulSets": kube.list("statefulsets"),
            "DaemonSets": kube.list("daemonsets"),
            "ReplicationControllers": kube.list("replicationcontrollers"),
            "persistent volume claims": kube.list("persistentvolumeclaims"),
            "Jobs": kube.list("jobs"),
            "CronJobs": kube.list("cronjobs"),
        }
        present = [label for label, items in forbidden.items() if items]
        require(not present, f"forbidden workload/storage resources exist: {present}")
        require(all(item.get("spec", {}).get("clusterIP") != "None" for item in kube.list("services")), "headless Services are forbidden")
        return "exactly the backend and gateway Deployments and Services exist"

    def r03() -> str:
        config = kube.get("configmap", "backend-content")
        require(config.get("data") == {"ready": "ready\n", "value": "42\n"}, f"backend-content data differs: {config.get('data')}")
        deployment, container = named_main("backend", "backend")
        assert_configmap_mount(deployment, container, "backend-content", "/srv")
        return "exact backend content is mounted read-only at /srv"

    def r04() -> str:
        config = kube.get("configmap", "gateway-content")
        require(config.get("data") == {"index.html": "Gateway online\n", "health": "ok\n"}, f"gateway-content data differs: {config.get('data')}")
        deployment, container = named_main("gateway", "gateway")
        assert_configmap_mount(deployment, container, "gateway-content", "/www")
        secret = kube.get("secret", "gateway-secret")
        secret_data = secret.get("data", {})
        require(set(secret_data) == {"BACKEND_TOKEN"}, f"gateway-secret keys differ: {sorted(secret_data)}")
        require(_decode_secret_value(secret_data.get("BACKEND_TOKEN", ""), "BACKEND_TOKEN") == "pilot-token", "BACKEND_TOKEN placeholder differs")
        token_env = [item for item in container.get("env", []) or [] if item.get("name") == "BACKEND_TOKEN"]
        require(len(token_env) == 1, "gateway main container must define BACKEND_TOKEN exactly once")
        token_ref = token_env[0].get("valueFrom", {}).get("secretKeyRef", {})
        require(token_ref.get("name") == "gateway-secret" and token_ref.get("key") == "BACKEND_TOKEN", "BACKEND_TOKEN does not reference gateway-secret/BACKEND_TOKEN")
        require(not container.get("envFrom"), "gateway main container must not expose gateway-secret through envFrom")
        init_containers = pod_spec(deployment).get("initContainers", [])
        _assert_secret_not_in_environment({"containers": init_containers}, "gateway-secret")
        _assert_no_secret_volume(pod_spec(deployment), "gateway-secret")
        return "gateway content and environment-only Secret are wired exactly"

    def r05() -> str:
        deployment, container = named_main("backend", "backend")
        require(deployment.get("spec", {}).get("replicas") == 1, "backend must have one replica")
        require(pod_labels(deployment).get("app") == "backend", "backend app label is missing")
        _assert_pod_label_and_service_account(deployment, "backend", "backend-sa")
        require(container.get("image") == "busybox:1.36.1", "backend image differs")
        require([item.get("containerPort") for item in container.get("ports", [])] == [8081], "backend must expose only container port 8081")
        text = command_text(container)
        for value in ("httpd", "-f", "8081", "/srv"):
            require(value in text, f"backend command does not reference {value}")
        assert_http_probes(container, "/ready", 8081)
        service = kube.get("service", "backend-svc")
        require(service.get("spec", {}).get("type", "ClusterIP") == "ClusterIP", "backend-svc is not ClusterIP")
        require(service.get("spec", {}).get("selector", {}).get("app") == "backend", "backend-svc selector differs")
        require(len(service.get("spec", {}).get("ports", [])) == 1, "backend-svc must expose exactly one port")
        port = _service_port(service, 8081)
        require(port.get("protocol", "TCP") == "TCP" and _resolved_target_port(port, deployment) == "8081", "backend-svc port mapping differs")
        kube.rollout("deployment", "backend", timeout_seconds=120)
        return "backend identity, probes, command, and Service mapping are correct"

    def r06() -> str:
        deployment, container = named_main("gateway", "gateway")
        require(deployment.get("spec", {}).get("replicas") == 2, "gateway must have two replicas")
        require(pod_labels(deployment).get("app") == "gateway", "gateway app label is missing")
        _assert_pod_label_and_service_account(deployment, "gateway", "gateway-sa")
        require(container.get("image") == "busybox:1.36.1", "gateway image differs")
        require([item.get("containerPort") for item in container.get("ports", [])] == [8080], "gateway must expose only container port 8080")
        text = command_text(container)
        for value in ("httpd", "-f", "8080", "/www"):
            require(value in text, f"gateway command does not reference {value}")
        assert_http_probes(container, "/health", 8080)
        strategy = deployment.get("spec", {}).get("strategy", {})
        require(strategy.get("type", "RollingUpdate") == "RollingUpdate", "gateway must use RollingUpdate")
        rolling = strategy.get("rollingUpdate", {})
        require(int_or_string(rolling.get("maxUnavailable")) == "0", "gateway maxUnavailable must be 0")
        require(int_or_string(rolling.get("maxSurge")) == "1", "gateway maxSurge must be 1")
        service = kube.get("service", "gateway-svc")
        require(service.get("spec", {}).get("type", "ClusterIP") == "ClusterIP", "gateway-svc is not ClusterIP")
        require(service.get("spec", {}).get("selector", {}).get("app") == "gateway", "gateway-svc selector differs")
        require(len(service.get("spec", {}).get("ports", [])) == 1, "gateway-svc must expose exactly one port")
        port = _service_port(service, 80)
        require(port.get("protocol", "TCP") == "TCP" and _resolved_target_port(port, deployment) == "8080", "gateway-svc port mapping differs")
        kube.rollout("deployment", "gateway", timeout_seconds=120)
        return "gateway identity, probes, rollout, command, and Service mapping are correct"

    def r07() -> str:
        deployment = kube.get("deployment", "gateway")
        init_containers = pod_spec(deployment).get("initContainers", [])
        require(len(init_containers) == 1, "gateway must have exactly one init container")
        init = init_containers[0]
        require(init.get("name") == "wait-backend", "init container must be wait-backend")
        require(init.get("image") == "busybox:1.36.1", "wait-backend image differs")
        text = command_text(init)
        for value in ("http://backend-svc:8081/ready", "ready", "sleep"):
            require(value in text, f"wait-backend command does not reference {value}")
        lowered = text.lower()
        require("|| true" not in lowered and "while true" not in lowered and "while :" not in lowered, "wait-backend suppresses failure or loops forever")
        require(
            _has_bounded_retry_and_nonzero_failure(
                lowered, maximum_attempts=30
            ),
            "wait-backend does not expose a recognizable at-most-30-attempt bound and non-zero terminal failure",
        )
        gateway_pods = kube.list("pods", "app=gateway")
        require(len(gateway_pods) == 2, f"expected two gateway pods, got {len(gateway_pods)}")
        for pod in gateway_pods:
            statuses = pod.get("status", {}).get("initContainerStatuses", [])
            require(len(statuses) == 1, "gateway pod init status count differs")
            terminated = statuses[0].get("state", {}).get("terminated", {})
            require(terminated.get("exitCode") == 0, f"wait-backend did not terminate successfully: {terminated}")
        return "bounded wait-backend dependency gate completed successfully"

    def r08() -> str:
        accounts = {item.get("metadata", {}).get("name") for item in kube.list("serviceaccounts")}
        require({"backend-sa", "gateway-sa"} <= accounts, "required ServiceAccounts are missing")
        for workload_name, account in (("backend", "backend-sa"), ("gateway", "gateway-sa")):
            spec = pod_spec(kube.get("deployment", workload_name))
            require(spec.get("serviceAccountName") == account, f"{workload_name} ServiceAccount differs")
            require(spec.get("automountServiceAccountToken") is False, f"{workload_name} must disable token automount")
        require(not kube.list("roles") and not kube.list("rolebindings"), "no Roles or RoleBindings are allowed")
        return "separate ServiceAccounts are used without token mounting or namespaced RBAC"

    def r09() -> str:
        for workload_name in ("backend", "gateway"):
            deployment = kube.get("deployment", workload_name)
            assert_hardened(deployment, workload_name)
            for container in all_containers(pod_spec(deployment)):
                require(container.get("securityContext", {}).get("allowPrivilegeEscalation") is False, f"{workload_name} container permits privilege escalation")
        return "all main and init containers use the required hardening"

    def r10() -> str:
        names = {item.get("metadata", {}).get("name") for item in kube.list("networkpolicies")}
        expected = {"default-deny-apps", "allow-gateway-backend", "allow-gateway-egress", "allow-evaluator-ingress"}
        require(names == expected, f"NetworkPolicy names differ: {sorted(names)}")
        deny = policy("default-deny-apps")
        require(has_role_expression(deny, {"backend", "gateway"}), "default-deny-apps does not select both roles")
        require(set(deny.get("spec", {}).get("policyTypes", [])) == {"Ingress", "Egress"}, "default-deny-apps policyTypes differ")
        require(deny.get("spec", {}).get("ingress", []) == [] and deny.get("spec", {}).get("egress", []) == [], "default-deny-apps is not a complete default deny")
        return "exactly four policies exist and default-deny selects both applications"

    def r11() -> str:
        ingress = policy("allow-gateway-backend")
        require(
            selector_labels(ingress) == {"role": "backend"},
            "allow-gateway-backend target differs",
        )
        ingress_rules = ingress.get("spec", {}).get("ingress", [])
        require(
            len(ingress_rules) == 1
            and _network_policy_rule_has_exact_ports(
                ingress_rules[0], {("TCP", "8081")}
            ),
            "allow-gateway-backend ports differ",
        )
        peers = ingress_rules[0].get("from", [])
        require(
            len(peers) == 1
            and peers[0].get("podSelector", {}).get("matchLabels")
            == {"role": "gateway"}
            and set(peers[0]) == {"podSelector"},
            "allow-gateway-backend source differs",
        )

        egress = policy("allow-gateway-egress")
        require(
            selector_labels(egress) == {"role": "gateway"},
            "allow-gateway-egress target differs",
        )
        rules = egress.get("spec", {}).get("egress", [])
        backend_rules = [
            rule
            for rule in rules
            if _network_policy_rule_has_exact_ports(rule, {("TCP", "8081")})
            and len(rule.get("to", [])) == 1
            and rule.get("to", [])[0]
            .get("podSelector", {})
            .get("matchLabels")
            == {"role": "backend"}
            and set(rule.get("to", [])[0]) == {"podSelector"}
        ]
        dns_rules = [
            rule
            for rule in rules
            if _network_policy_rule_has_exact_ports(
                rule, {("UDP", "53"), ("TCP", "53")}
            )
        ]
        require(
            len(rules) == 2 and len(backend_rules) == 1,
            "gateway backend egress rule differs",
        )
        backend_peers = backend_rules[0].get("to", [])
        require(
            len(backend_peers) == 1
            and backend_peers[0].get("podSelector", {}).get("matchLabels")
            == {"role": "backend"}
            and set(backend_peers[0]) == {"podSelector"},
            "gateway backend egress peer is broader than role=backend in this namespace",
        )
        require(len(dns_rules) == 1, "gateway DNS egress rule differs")

        evaluator = policy("allow-evaluator-ingress")
        require(
            has_role_expression(evaluator, {"backend", "gateway"}),
            "allow-evaluator-ingress targets differ",
        )
        evaluator_rules = evaluator.get("spec", {}).get("ingress", [])
        require(
            len(evaluator_rules) == 1
            and _network_policy_rule_has_exact_ports(
                evaluator_rules[0], {("TCP", "8080"), ("TCP", "8081")}
            ),
            "evaluator ingress ports differ",
        )
        evaluator_peers = evaluator_rules[0].get("from", [])
        require(
            len(evaluator_peers) == 1
            and evaluator_peers[0].get("podSelector", {}).get("matchLabels")
            == {"access": "probe"}
            and set(evaluator_peers[0]) == {"podSelector"},
            "evaluator ingress source differs",
        )
        return "gateway/backend, DNS, and evaluator allow rules are narrowly scoped"

    def r12() -> str:
        gateway_pods = kube.list("pods", "app=gateway")
        backend_pods = kube.list("pods", "app=backend")
        require(gateway_pods and backend_pods, "application pods are missing")
        gateway_name = gateway_pods[0].get("metadata", {}).get("name", "")
        backend_name = backend_pods[0].get("metadata", {}).get("name", "")
        try:
            kube.exec(gateway_name, "test \"$(wget -q -T 5 -O - http://backend-svc:8081/ready)\" = ready", "gateway")
        except RequirementFailure as exc:
            raise RequirementFailure(
                "gateway could not resolve backend-svc and fetch its exact ready response"
            ) from exc
        allowed = kube.run_probe_pod(
            "aipc-eval-authorized",
            "test \"$(wget -q -T 5 -O - http://backend-svc:8081/value)\" = 42; test \"$(wget -q -T 5 -O - http://gateway-svc:80/)\" = \"Gateway online\"; test \"$(wget -q -T 5 -O - http://gateway-svc:80/health)\" = ok",
            labels={"access": "probe"},
        )
        blocked_ingress = kube.run_probe_pod(
            "aipc-eval-untrusted",
            "wget -q -T 4 -O - http://backend-svc:8081/ready",
            labels={"access": "untrusted"},
            expect_success=False,
        )
        kube.exec(backend_name, "if wget -q -T 4 -O - http://gateway-svc:80/health; then exit 1; else exit 0; fi", "backend")
        return f"gateway resolved and reached backend-svc; authorized {allowed}; untrusted {blocked_ingress}; backend egress blocked"

    def r13() -> str:
        pdb = kube.get("poddisruptionbudget", "gateway-pdb")
        require(pdb.get("spec", {}).get("selector", {}).get("matchLabels", {}).get("role") == "gateway", "gateway-pdb selector differs")
        require(int_or_string(pdb.get("spec", {}).get("minAvailable")) == "1", "gateway-pdb minAvailable differs")
        quota = kube.get("resourcequota", "service-chain-quota")
        hard = quota.get("status", {}).get("hard", quota.get("spec", {}).get("hard", {}))
        require(hard.get("pods") == "6", f"pod quota must be 6, got {hard.get('pods')}")
        backend_replicas = int(kube.get("deployment", "backend").get("spec", {}).get("replicas", 1))
        gateway = kube.get("deployment", "gateway")
        gateway_replicas = int(gateway.get("spec", {}).get("replicas", 1))
        surge = int_or_string(gateway.get("spec", {}).get("strategy", {}).get("rollingUpdate", {}).get("maxSurge", 1))
        require(surge.isdigit(), "gateway maxSurge must be an integer for quota-fit analysis")
        peak = backend_replicas + gateway_replicas + int(surge) + 1
        require(peak <= 6, f"rollout plus evaluator probe needs {peak} pods, exceeding quota 6")
        return f"gateway PDB is minAvailable 1 and computed rollout-plus-probe peak is {peak}/6"

    return [
        ("R01", "isolated service-chain namespace", lambda: _check_namespace(kube, "service-chain", _require_candidate_path(candidate_path))),
        ("R02", "exact two-service application topology", r02),
        ("R03", "exact backend content and mount", r03),
        ("R04", "exact gateway content and environment-only Secret", r04),
        ("R05", "backend Deployment, probes, and Service", r05),
        ("R06", "gateway Deployment, probes, rollout, and Service", r06),
        ("R07", "bounded backend dependency gate", r07),
        ("R08", "token-free ServiceAccounts and no RBAC", r08),
        ("R09", "container security contexts", r09),
        ("R10", "exact default-deny policy set", r10),
        ("R11", "narrow application and evaluator allow rules", r11),
        ("R12", "live service and network isolation behavior", r12),
        ("R13", "gateway disruption budget and quota fit", r13),
    ]


def _combined_command(container: dict[str, Any]) -> list[str]:
    return [*container.get("command", []), *container.get("args", [])]


def _single_named_container(workload: dict[str, Any], name: str) -> dict[str, Any]:
    containers = pod_spec(workload).get("containers", [])
    require(len(containers) == 1, "workload must contain exactly one main container")
    require(containers[0].get("name") == name, f"container must be named {name}")
    return containers[0]


def _assert_tcp_port(container: dict[str, Any], port: int, name: str | None = None) -> None:
    matches = [
        item
        for item in container.get("ports", [])
        if item.get("containerPort") == port
        and item.get("protocol", "TCP") == "TCP"
        and (name is None or item.get("name") == name)
    ]
    require(len(matches) == 1, f"container must expose the required TCP port {port}")


def _assert_tcp_readiness(
    container: dict[str, Any], port: int | str, period: int, failure: int | None = None
) -> None:
    probe = container.get("readinessProbe", {})
    require(probe.get("tcpSocket", {}).get("port") == port, "readiness probe targets the wrong TCP port")
    require(probe.get("periodSeconds", 10) == period, "readiness periodSeconds differs")
    if failure is not None:
        require(probe.get("failureThreshold", 3) == failure, "readiness failureThreshold differs")


def _assert_exact_body_probe(
    kube: Kubectl,
    name: str,
    url: str,
    expected: str,
    byte_count: int,
    *,
    labels: dict[str, str] | None = None,
) -> str:
    command = (
        f"wget -q -T 5 -O /tmp/body {url}; "
        f"test \"$(wc -c </tmp/body | tr -d ' ')\" = {byte_count}; "
        f"test \"$(cat /tmp/body)\" = {expected}"
    )
    return kube.run_probe_pod(name, command, labels=labels, timeout_seconds=35)


def _assert_cronjob(
    cronjob: dict[str, Any], schedule: str, expected_command: list[str]
) -> None:
    spec = cronjob.get("spec", {})
    require(spec.get("schedule") == schedule, "CronJob schedule differs")
    require(spec.get("suspend") is True, "CronJob must be suspended")
    require(spec.get("concurrencyPolicy") == "Forbid", "CronJob concurrencyPolicy must be Forbid")
    pod = _cron_spec(cronjob)
    require(pod.get("restartPolicy") == "Never", "CronJob restartPolicy must be Never")
    containers = pod.get("containers", [])
    require(len(containers) == 1, "CronJob must have exactly one container")
    require(containers[0].get("image") == "busybox:1.36.1", "CronJob image differs")
    require(_combined_command(containers[0]) == expected_command, "CronJob command differs")


def _assert_http_readiness(
    container: dict[str, Any],
    *,
    path: str,
    port: int | str,
    period: int,
    failure: int | None = None,
    initial_delay: int | None = None,
) -> None:
    probe = container.get("readinessProbe", {})
    http_get = probe.get("httpGet", {})
    require(http_get.get("path") == path, "readiness probe path differs")
    require(http_get.get("port") == port, "readiness probe port differs")
    require(probe.get("periodSeconds", 10) == period, "readiness periodSeconds differs")
    if failure is not None:
        require(probe.get("failureThreshold", 3) == failure, "readiness failureThreshold differs")
    if initial_delay is not None:
        require(probe.get("initialDelaySeconds", 0) == initial_delay, "readiness initialDelaySeconds differs")


def _assert_medium_hardening(workload: dict[str, Any], label: str) -> None:
    spec = pod_spec(workload)
    pod_security = spec.get("securityContext", {})
    expected_pod = {
        "runAsNonRoot": True,
        "runAsUser": 1000,
        "runAsGroup": 1000,
        "fsGroup": 1000,
    }
    for field, expected in expected_pod.items():
        require(pod_security.get(field) == expected, f"{label}: pod securityContext {field} differs")
    for container in all_containers(spec):
        security = container.get("securityContext", {})
        require(security.get("allowPrivilegeEscalation") is False, f"{label}: allowPrivilegeEscalation must be false")
        require(security.get("readOnlyRootFilesystem") is True, f"{label}: readOnlyRootFilesystem must be true")
        require(set(security.get("capabilities", {}).get("drop", [])) == {"ALL"}, f"{label}: capabilities.drop must be exactly ALL")


def _assert_single_ingress_policy(
    policy: dict[str, Any], *, app: str, access: str, port: int
) -> None:
    spec = policy.get("spec", {})
    require(spec.get("podSelector", {}).get("matchLabels") == {"app": app}, "NetworkPolicy selector differs")
    require(set(spec.get("policyTypes", ["Ingress"])) == {"Ingress"}, "policyTypes must contain only Ingress")
    ingress = spec.get("ingress", [])
    require(len(ingress) == 1, "NetworkPolicy must contain exactly one ingress rule")
    require(
        ingress[0].get("from", []) == [{"podSelector": {"matchLabels": {"access": access}}}],
        f"ingress source must be the same-namespace access={access} selector",
    )
    require(
        _network_policy_rule_has_exact_ports(ingress[0], {("TCP", str(port))}),
        "NetworkPolicy port allowance differs",
    )


def _ready_pod(pod: dict[str, Any]) -> bool:
    return any(
        item.get("type") == "Ready" and item.get("status") == "True"
        for item in pod.get("status", {}).get("conditions", [])
    )


def _ready_endpoint_uids(endpoints: dict[str, Any]) -> set[str]:
    return {
        address.get("targetRef", {}).get("uid", "")
        for subset in endpoints.get("subsets", [])
        for address in subset.get("addresses", [])
        if address.get("targetRef", {}).get("uid")
    }


def _all_ready_pods_serve_base64(
    kube: Kubectl,
    label: str,
    container: str,
    expected_count: int,
    expected_base64: str,
) -> bool:
    pods = [pod for pod in kube.list("pods", label) if _ready_pod(pod)]
    if len(pods) != expected_count:
        return False
    for pod in pods:
        name = pod.get("metadata", {}).get("name", "")
        result = kube.run(
            [
                "exec",
                "-n",
                kube.namespace,
                name,
                "-c",
                container,
                "--",
                "sh",
                "-ec",
                f'test "$(wget -qO- http://127.0.0.1:8080/ | base64 | tr -d \'\\n\')" = {expected_base64}',
            ],
            check=False,
        )
        if result.returncode != 0:
            return False
    return True


def _assert_exact_pdb(kube: Kubectl, name: str, app: str) -> str:
    pdb = kube.get("poddisruptionbudget", name)
    require(pdb.get("spec", {}).get("selector", {}).get("matchLabels") == {"app": app}, f"{name} selector differs")
    require(int_or_string(pdb.get("spec", {}).get("minAvailable")) == "1", f"{name} minAvailable must be 1")
    return f"{name} selects app={app} with minAvailable 1"


def _raise_candidate_mutation_failure(
    kube: Kubectl, exc: KubectlError, description: str
) -> None:
    detail = f"{exc.stdout}\n{exc.stderr}".lower()
    candidate_markers = ("immutable", "forbidden", "denied", "invalid", "not found", "notfound")
    if not any(marker in detail for marker in candidate_markers):
        raise exc
    ready = kube.run(["get", "--raw=/readyz"], check=False, timeout=min(kube.command_timeout, 10))
    if ready.returncode == 0 and "ok" in ready.stdout.lower():
        raise RequirementFailure(
            f"the evaluator could not perform {description} because the candidate rejected it while the Kubernetes API remained healthy: {exc}"
        ) from exc
    raise exc


def _run_cronjob_and_assert_log(
    kube: Kubectl, cronjob: str, suffix: str, expected_log: str, timeout: int = 60
) -> str:
    name = f"aipc-eval-{suffix}"
    kube.delete("job", name)
    try:
        try:
            kube.run(["create", "job", name, f"--from=cronjob/{cronjob}", "-n", kube.namespace])
        except KubectlError as exc:
            kube._raise_evaluator_creation_failure(exc, f"probe Job {name}")
        terminal: dict[str, Any] = {}
        condition: str | None = None

        def completed() -> bool:
            nonlocal terminal, condition
            terminal = kube.get("job", name)
            condition = job_terminal_condition(terminal)
            return condition is not None

        kube.wait_until(completed, f"Job {name} to reach a terminal condition", timeout, 1)
        require(condition == "Complete", f"{name} did not complete successfully: {terminal.get('status', {})}")
        pods = kube.list("pods", f"job-name={name}")
        require(len(pods) == 1, f"{name} must have exactly one pod, found {len(pods)}")
        pod_name = pods[0].get("metadata", {}).get("name", "")
        logs = kube.run(["logs", pod_name, "-n", kube.namespace]).stdout
        require(logs == expected_log, f"{name} logs differ: {logs!r}")
        return f"Job {name} completed and logged exactly {expected_log.rstrip()!r}"
    finally:
        kube.delete("job", name)


def _easy1_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        deployment = kube.get("deployment", "status-web")
        require(deployment.get("spec", {}).get("replicas") == 1, "status-web must have one replica")
        require(pod_labels(deployment).get("app") == "status-web", "pod label app=status-web is missing")
        container = _single_named_container(deployment, "web")
        require(container.get("image") == "busybox:1.36.1", "web image differs")
        expected = [
            "sh",
            "-c",
            "mkdir -p /www && printf 'status=green\\n' > /www/index.html && httpd -f -p 8080 -h /www",
        ]
        require(_combined_command(container) == expected, "status-web command differs")
        _assert_tcp_port(container, 8080, "http")
        _assert_tcp_readiness(container, "http", 5, 3)
        kube.rollout("deployment", "status-web", timeout_seconds=60)
        return "status-web has the exact image, command, port, probe, label, and Ready replica"

    def r03() -> str:
        deployment = kube.get("deployment", "status-web")
        spec = deployment.get("spec", {})
        require(spec.get("strategy", {}).get("type") == "Recreate", "strategy must be Recreate")
        require(spec.get("revisionHistoryLimit") == 1, "revisionHistoryLimit must be 1")
        return "status-web uses Recreate with revisionHistoryLimit 1"

    def r04() -> str:
        service = kube.get("service", "status-svc")
        spec = service.get("spec", {})
        require(spec.get("type", "ClusterIP") == "ClusterIP", "status-svc is not ClusterIP")
        require(spec.get("selector") == {"app": "status-web"}, "status-svc selector differs")
        port = _service_port(service, 80)
        require(port.get("protocol", "TCP") == "TCP", "status-svc port is not TCP")
        require(_resolved_target_port(port, kube.get("deployment", "status-web")) == "8080", "targetPort does not resolve to 8080")
        cluster_ip = spec.get("clusterIP")
        require(cluster_ip not in {None, "", "None"}, "status-svc has no ClusterIP")
        probe = _assert_exact_body_probe(
            kube,
            "aipc-eval-status-allowed",
            f"http://{cluster_ip}:80/",
            "status=green",
            13,
            labels={"access": "status"},
        )
        return f"status-svc returns the exact body to the approved client; {probe}"

    def r05() -> str:
        policy = kube.get("networkpolicy", "status-ingress")
        spec = policy.get("spec", {})
        require(spec.get("podSelector", {}).get("matchLabels") == {"app": "status-web"}, "policy selector differs")
        require(set(spec.get("policyTypes", ["Ingress"])) == {"Ingress"}, "policyTypes must contain only Ingress")
        ingress = spec.get("ingress", [])
        require(len(ingress) == 1, "status-ingress must have exactly one ingress rule")
        peers = ingress[0].get("from", [])
        require(len(peers) == 1, "status-ingress must have exactly one ingress peer")
        require(peers[0] == {"podSelector": {"matchLabels": {"access": "status"}}}, "ingress peer must be the same-namespace access=status selector")
        require(_network_policy_rule_has_exact_ports(ingress[0], {("TCP", "8080")}), "ingress port allowance differs")
        service = kube.get("service", "status-svc")
        cluster_ip = service.get("spec", {}).get("clusterIP")
        allowed = kube.run_probe_pod(
            "aipc-eval-policy-allowed",
            f"wget -q -T 5 -O /dev/null http://{cluster_ip}:80/",
            labels={"access": "status"},
            timeout_seconds=35,
        )
        denied = kube.run_probe_pod(
            "aipc-eval-policy-denied",
            f"wget -q -T 5 -O /dev/null http://{cluster_ip}:80/",
            labels={"access": "other"},
            expect_success=False,
            timeout_seconds=35,
        )
        return f"approved ingress succeeds and unapproved ingress fails; {allowed}; {denied}"

    def r06() -> str:
        _assert_cronjob(
            kube.get("cronjob", "diagnostics-report"),
            "0 * * * *",
            ["sh", "-c", "echo diagnostics-ready"],
        )
        return "diagnostics-report is the exact suspended hourly CronJob"

    return [
        ("R01", "isolated easy-status namespace", lambda: _check_namespace(kube, "easy-status-ns", _require_candidate_path(candidate_path))),
        ("R02", "status Deployment and readiness", r02),
        ("R03", "status Deployment strategy", r03),
        ("R04", "exact status Service response", r04),
        ("R05", "status ingress isolation", r05),
        ("R06", "suspended diagnostics CronJob", r06),
    ]


def _easy2_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        config = kube.get("configmap", "calc-inputs")
        require(config.get("data") == {"LEFT": "7", "RIGHT": "5"}, "calc-inputs data differs")
        return "calc-inputs contains exactly LEFT=7 and RIGHT=5"

    def r03() -> str:
        secret = kube.get("secret", "calc-operation")
        require(secret.get("type") == "Opaque", "calc-operation must be Opaque")
        data = secret.get("data", {})
        require(set(data) == {"OPERATION"}, "calc-operation must contain only OPERATION")
        require(_decode_secret_value(data.get("OPERATION"), "OPERATION") == "add", "OPERATION differs")
        return "calc-operation is the exact one-key Opaque Secret"

    def r04() -> str:
        job = kube.get("job", "sum-job")
        spec = job.get("spec", {})
        pod = spec.get("template", {}).get("spec", {})
        require(pod.get("restartPolicy") == "Never", "sum-job restartPolicy must be Never")
        containers = pod.get("containers", [])
        require(len(containers) == 1, "sum-job must have exactly one container")
        container = containers[0]
        require(container.get("image") == "busybox:1.36.1", "sum-job image differs")
        require(
            _combined_command(container)
            == [
                "sh",
                "-c",
                'test "$OPERATION" = add && result=$((LEFT + RIGHT)) && echo result=$result && test "$result" -eq 12',
            ],
            "sum-job command differs",
        )
        require(_container_exposes_configmap_keys(container, "calc-inputs", {"LEFT", "RIGHT"}), "LEFT and RIGHT are not sourced from calc-inputs")
        operation_env = [item for item in container.get("env", []) if item.get("name") == "OPERATION"]
        require(len(operation_env) == 1, "sum-job must define OPERATION exactly once")
        require(
            operation_env[0].get("valueFrom", {}).get("secretKeyRef", {})
            == {"name": "calc-operation", "key": "OPERATION"},
            "OPERATION is not sourced through the required secretKeyRef",
        )
        require(
            not any(item.get("secretRef", {}).get("name") == "calc-operation" for item in container.get("envFrom", [])),
            "calc-operation must not be exposed through envFrom",
        )
        require(spec.get("backoffLimit") == 1, "backoffLimit must be 1")
        require(spec.get("activeDeadlineSeconds") == 60, "activeDeadlineSeconds must be 60")
        kube.wait_until(
            lambda: kube.get("job", "sum-job").get("status", {}).get("succeeded", 0) == 1,
            "sum-job to complete once",
            60,
            1,
        )
        pods = kube.list("pods", "job-name=sum-job")
        require(len(pods) == 1, f"sum-job must have exactly one pod, found {len(pods)}")
        pod_name = pods[0].get("metadata", {}).get("name", "")
        logs = kube.run(["logs", pod_name, "-n", kube.namespace]).stdout
        require(logs == "result=12\n", f"sum-job logs differ: {logs!r}")
        return "sum-job uses the required projections, completes once, and logs exactly result=12"

    def r05() -> str:
        quota = kube.get("resourcequota", "batch-quota")
        hard = quota.get("spec", {}).get("hard", {})
        require(set(hard) == {"count/jobs.batch"}, "batch-quota contains missing or additional limits")
        require(int_or_string(hard.get("count/jobs.batch")) == "2", "count/jobs.batch must be 2")
        return "batch-quota limits batch/v1 Jobs to exactly two"

    def r06() -> str:
        spec = kube.get("job", "sum-job").get("spec", {})
        require(spec.get("completions") == 1, "completions must be 1")
        require(spec.get("parallelism") == 1, "parallelism must be 1")
        require(spec.get("ttlSecondsAfterFinished") == 600, "ttlSecondsAfterFinished must be 600")
        return "sum-job has the exact completion, parallelism, and TTL values"

    return [
        ("R01", "isolated easy-calc namespace", lambda: _check_namespace(kube, "easy-calc-ns", _require_candidate_path(candidate_path))),
        ("R02", "exact calculation ConfigMap", r02),
        ("R03", "exact operation Secret", r03),
        ("R04", "calculation Job completion and output", r04),
        ("R05", "batch Job quota", r05),
        ("R06", "calculation Job lifecycle", r06),
    ]


def _easy3_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        config = kube.get("configmap", "audit-target")
        require(config.get("data") == {"MODE": "read-only"}, "audit-target data differs")
        return "audit-target contains exactly MODE=read-only"

    def r03() -> str:
        config = kube.get("configmap", "audit-target")
        require(config.get("immutable") is True, "audit-target must be immutable")
        return "audit-target is immutable"

    def r04() -> str:
        account = kube.get("serviceaccount", "audit-reader")
        require(account.get("metadata", {}).get("name") == "audit-reader", "audit-reader is absent")
        return "audit-reader ServiceAccount exists"

    def r05() -> str:
        role = kube.get("role", "audit-reader-role")
        rules = role.get("rules", [])
        require(len(rules) == 1, "audit-reader-role must contain exactly one rule")
        rule = rules[0]
        require(rule.get("apiGroups") == [""], "Role apiGroups must be exactly ['']")
        require(rule.get("resources") == ["configmaps"], "Role resources must be exactly ['configmaps']")
        require(rule.get("resourceNames") == ["audit-target"], "Role resourceNames must be exactly ['audit-target']")
        require(rule.get("verbs") == ["get"], "Role verbs must be exactly ['get']")
        return "audit-reader-role grants only get on ConfigMap audit-target"

    def r06() -> str:
        binding = kube.get("rolebinding", "audit-reader-binding")
        subjects = binding.get("subjects", [])
        require(
            subjects
            == [
                {
                    "kind": "ServiceAccount",
                    "name": "audit-reader",
                    "namespace": "easy-rbac-ns",
                }
            ],
            "RoleBinding must contain only the audit-reader ServiceAccount subject",
        )
        role_ref = binding.get("roleRef", {})
        require(
            role_ref
            == {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "audit-reader-role",
            },
            "RoleBinding roleRef differs",
        )
        require(
            kube.auth_can_i("audit-reader", "get", "configmap/audit-target"),
            "audit-reader is not initially authorized",
        )
        restore = {
            "apiVersion": binding.get("apiVersion", "rbac.authorization.k8s.io/v1"),
            "kind": "RoleBinding",
            "metadata": {
                "name": "audit-reader-binding",
                "namespace": "easy-rbac-ns",
            },
            "roleRef": role_ref,
            "subjects": subjects,
        }
        kube.run(
            [
                "delete",
                "rolebinding",
                "audit-reader-binding",
                "-n",
                kube.namespace,
                "--wait=true",
            ]
        )
        try:
            kube.wait_until(
                lambda: not kube.auth_can_i("audit-reader", "get", "configmap/audit-target"),
                "audit-reader authorization to become denied",
                15,
                1,
            )
        finally:
            kube.run(
                ["apply", "-f", "-"],
                input_text=json.dumps(restore),
            )
        kube.wait_until(
            lambda: kube.auth_can_i("audit-reader", "get", "configmap/audit-target"),
            "audit-reader authorization to recover",
            15,
            1,
        )
        return "audit-reader authorization follows the required yes-no-yes transition"

    return [
        ("R01", "isolated easy-rbac namespace", lambda: _check_namespace(kube, "easy-rbac-ns", _require_candidate_path(candidate_path))),
        ("R02", "exact RBAC target ConfigMap", r02),
        ("R03", "immutable RBAC target", r03),
        ("R04", "audit-reader ServiceAccount", r04),
        ("R05", "single-object least-privilege Role", r05),
        ("R06", "RoleBinding authorization loss and recovery", r06),
    ]


def _easy4_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        secret = kube.get("secret", "banner-secret")
        require(secret.get("type") == "Opaque", "banner-secret must be Opaque")
        data = secret.get("data", {})
        require(set(data) == {"BANNER"}, "banner-secret must contain only BANNER")
        require(_decode_secret_value(data.get("BANNER"), "BANNER") == "hello-restart", "BANNER differs")
        return "banner-secret is the exact one-key Opaque Secret"

    def r03() -> str:
        require(kube.get("secret", "banner-secret").get("immutable") is True, "banner-secret must be immutable")
        return "banner-secret is immutable"

    def r04() -> str:
        deployment = kube.get("deployment", "restart-web")
        require(deployment.get("spec", {}).get("replicas") == 1, "restart-web must have one replica")
        require(pod_labels(deployment).get("app") == "restart-web", "pod label app=restart-web is missing")
        container = _single_named_container(deployment, "web")
        require(container.get("image") == "busybox:1.36.1", "web image differs")
        require(
            _combined_command(container)
            == [
                "sh",
                "-c",
                "mkdir -p /work && printf 'ready\\n' > /work/index.html && httpd -f -p 8080 -h /work",
            ],
            "restart-web command differs",
        )
        banner_env = [item for item in container.get("env", []) if item.get("name") == "BANNER"]
        require(len(banner_env) == 1, "restart-web must define BANNER exactly once")
        require(
            banner_env[0].get("valueFrom", {}).get("secretKeyRef", {})
            == {"name": "banner-secret", "key": "BANNER"},
            "BANNER is not sourced through the required secretKeyRef",
        )
        require(
            not any(item.get("secretRef", {}).get("name") == "banner-secret" for item in container.get("envFrom", [])),
            "banner-secret must not be exposed through envFrom",
        )
        _assert_tcp_port(container, 8080)
        _assert_tcp_readiness(container, 8080, 5)
        kube.rollout("deployment", "restart-web", timeout_seconds=60)
        return "restart-web has the exact Secret projection, command, port, probe, and Ready replica"

    def r05() -> str:
        deployment = kube.get("deployment", "restart-web")
        service = kube.get("service", "restart-svc")
        spec = service.get("spec", {})
        require(spec.get("type", "ClusterIP") == "ClusterIP", "restart-svc is not ClusterIP")
        require(spec.get("selector") == {"app": "restart-web"}, "restart-svc selector differs")
        port = _service_port(service, 8080)
        require(port.get("protocol", "TCP") == "TCP", "restart-svc port is not TCP")
        require(_resolved_target_port(port, deployment) == "8080", "restart-svc targetPort differs")
        pods = kube.list("pods", "app=restart-web")
        require(len(pods) == 1, f"expected one initial restart-web pod, found {len(pods)}")
        original_name = pods[0].get("metadata", {}).get("name", "")
        original_uid = pods[0].get("metadata", {}).get("uid", "")
        require(bool(original_name and original_uid), "initial restart-web pod has no stable identity")
        kube.run(["delete", "pod", original_name, "-n", kube.namespace, "--wait=true"])
        replacement: dict[str, Any] = {}

        def replacement_ready() -> bool:
            nonlocal replacement
            candidates = kube.list("pods", "app=restart-web")
            for candidate in candidates:
                uid = candidate.get("metadata", {}).get("uid")
                ready = any(
                    item.get("type") == "Ready" and item.get("status") == "True"
                    for item in candidate.get("status", {}).get("conditions", [])
                )
                if uid and uid != original_uid and ready:
                    replacement = candidate
                    return True
            return False

        kube.wait_until(replacement_ready, "a distinct Ready restart-web replacement pod", 60, 1)
        cluster_ip = spec.get("clusterIP")
        require(cluster_ip not in {None, "", "None"}, "restart-svc has no ClusterIP")
        probe = _assert_exact_body_probe(
            kube,
            "aipc-eval-restart-response",
            f"http://{cluster_ip}:8080/",
            "ready",
            6,
        )
        uid = replacement.get("metadata", {}).get("uid")
        return f"pod {original_uid} was replaced by {uid} and the exact Service response recovered; {probe}"

    def r06() -> str:
        quota = kube.get("resourcequota", "restart-quota")
        hard = quota.get("spec", {}).get("hard", {})
        require(set(hard) == {"pods", "services", "secrets"}, "restart-quota hard-limit keys differ")
        expected = {"pods": "5", "services": "2", "secrets": "3"}
        require({key: int_or_string(value) for key, value in hard.items()} == expected, "restart-quota values differ")
        return "restart-quota has exactly the three requested hard limits"

    return [
        ("R01", "isolated easy-restart namespace", lambda: _check_namespace(kube, "easy-restart-ns", _require_candidate_path(candidate_path))),
        ("R02", "exact banner Secret", r02),
        ("R03", "immutable banner Secret", r03),
        ("R04", "restart-web Deployment wiring", r04),
        ("R05", "pod replacement and Service recovery", r05),
        ("R06", "restart namespace quota", r06),
    ]


def _easy5_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        config = kube.get("configmap", "diagnostics-content")
        require(config.get("data") == {"index.html": "diagnostics-ready\n"}, "diagnostics-content data differs")
        return "diagnostics-content contains the exact index.html body"

    def r03() -> str:
        deployment = kube.get("deployment", "diagnostics-web")
        require(deployment.get("spec", {}).get("replicas") == 1, "diagnostics-web must have one replica")
        require(pod_labels(deployment).get("app") == "diagnostics-web", "pod label app=diagnostics-web is missing")
        container = _single_named_container(deployment, "web")
        require(container.get("image") == "busybox:1.36.1", "web image differs")
        require(_combined_command(container) == ["httpd", "-f", "-p", "8080", "-h", "/www"], "diagnostics-web command differs")
        _assert_tcp_port(container, 8080, "http")
        _assert_tcp_readiness(container, "http", 5)
        spec = pod_spec(deployment)
        mount = find_mount(container, "/www")
        require(mount is not None, "diagnostics-content is not mounted at /www")
        require(mount.get("readOnly") is True, "/www mount must be explicitly read-only")
        volume = find_volume(spec, mount.get("name", ""))
        require(volume is not None, "the /www volume does not exist")
        require(volume.get("configMap", {}).get("name") == "diagnostics-content", "/www is not backed by diagnostics-content")
        return "diagnostics-web has the exact ConfigMap mount, listener, port, and readiness probe"

    def r04() -> str:
        deployment = kube.get("deployment", "diagnostics-web")
        spec = pod_spec(deployment)
        pod_security = spec.get("securityContext", {})
        require(pod_security.get("runAsNonRoot") is True, "runAsNonRoot must be true")
        require(pod_security.get("runAsUser") == 1000, "runAsUser must be 1000")
        container = _single_named_container(deployment, "web")
        security = container.get("securityContext", {})
        require(security.get("allowPrivilegeEscalation") is False, "allowPrivilegeEscalation must be false")
        require(security.get("readOnlyRootFilesystem") is True, "readOnlyRootFilesystem must be true")
        require(set(security.get("capabilities", {}).get("drop", [])) == {"ALL"}, "capabilities.drop must be exactly ALL")
        return "diagnostics-web has every required pod and container hardening field"

    def r05() -> str:
        deployment = kube.get("deployment", "diagnostics-web")
        service = kube.get("service", "diagnostics-svc")
        spec = service.get("spec", {})
        require(spec.get("type", "ClusterIP") == "ClusterIP", "diagnostics-svc is not ClusterIP")
        require(spec.get("selector") == {"app": "diagnostics-web"}, "diagnostics-svc selector differs")
        port = _service_port(service, 80)
        require(port.get("protocol", "TCP") == "TCP", "diagnostics-svc port is not TCP")
        require(_resolved_target_port(port, deployment) == "8080", "targetPort does not resolve to 8080")
        kube.rollout("deployment", "diagnostics-web", timeout_seconds=60)
        endpoints = kube.get("endpoints", "diagnostics-svc")
        ready_addresses = [
            address
            for subset in endpoints.get("subsets", [])
            for address in subset.get("addresses", [])
        ]
        require(bool(ready_addresses), "diagnostics-svc has no Ready endpoint")
        cluster_ip = spec.get("clusterIP")
        require(cluster_ip not in {None, "", "None"}, "diagnostics-svc has no ClusterIP")
        probe = _assert_exact_body_probe(
            kube,
            "aipc-eval-diagnostics-response",
            f"http://{cluster_ip}:80/",
            "diagnostics-ready",
            18,
        )
        return f"diagnostics-svc has a Ready endpoint and returns the exact body; {probe}"

    def r06() -> str:
        _assert_cronjob(
            kube.get("cronjob", "diagnostics-check"),
            "*/10 * * * *",
            ["sh", "-c", "echo diagnostics-suspended"],
        )
        return "diagnostics-check is the exact suspended ten-minute CronJob"

    return [
        ("R01", "isolated easy-diagnostics namespace", lambda: _check_namespace(kube, "easy-diagnostics-ns", _require_candidate_path(candidate_path))),
        ("R02", "exact diagnostics content", r02),
        ("R03", "diagnostics Deployment wiring", r03),
        ("R04", "diagnostics pod hardening", r04),
        ("R05", "diagnostics endpoint and exact response", r05),
        ("R06", "suspended diagnostics check", r06),
    ]


def _medium1_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        config = kube.get("configmap", "portal-content")
        _assert_exact_configmap_data(config, {"index.html": "portal-version=v1\n"})
        return "portal-content contains exactly the initial portal body"

    def r03() -> str:
        secret = kube.get("secret", "portal-token")
        require(secret.get("type") == "Opaque", "portal-token must be Opaque")
        require(set(secret.get("data", {})) == {"API_TOKEN"}, "portal-token must contain only API_TOKEN")
        require(_decode_secret_value(secret.get("data", {}).get("API_TOKEN"), "API_TOKEN") == "medium-token", "API_TOKEN differs")
        return "portal-token is the exact one-key Opaque Secret"

    def r04() -> str:
        deployment = kube.get("deployment", "portal-web")
        require(deployment.get("spec", {}).get("replicas") == 2, "portal-web must have two replicas")
        require(pod_labels(deployment).get("app") == "portal-web", "pod label app=portal-web is missing")
        container = _single_named_container(deployment, "web")
        require(container.get("image") == "busybox:1.36.1", "portal web image differs")
        require(_combined_command(container) == ["sh", "-c", 'test "$API_TOKEN" = medium-token && httpd -f -p 8080 -h /www'], "portal web command differs")
        token_env = [item for item in container.get("env", []) if item.get("name") == "API_TOKEN"]
        require(len(token_env) == 1 and token_env[0].get("valueFrom", {}).get("secretKeyRef", {}) == {"name": "portal-token", "key": "API_TOKEN"}, "API_TOKEN is not sourced exactly once through portal-token/API_TOKEN secretKeyRef")
        require(not any(item.get("secretRef", {}).get("name") == "portal-token" for item in container.get("envFrom", [])), "portal-token must not be exposed through envFrom")
        mount = find_mount(container, "/www")
        require(mount is not None and mount.get("readOnly") is True, "/www must be mounted read-only")
        volume = find_volume(pod_spec(deployment), mount.get("name", ""))
        require(volume is not None and volume.get("configMap", {}).get("name") == "portal-content", "/www is not backed by portal-content")
        require(not volume.get("configMap", {}).get("items"), "portal-content must be mounted completely")
        _assert_tcp_port(container, 8080, "http")
        _assert_http_readiness(container, path="/", port="http", period=5, failure=3)
        kube.rollout("deployment", "portal-web", timeout_seconds=60)
        before = {pod.get("metadata", {}).get("uid") for pod in kube.list("pods", "app=portal-web")}
        patch_v2 = json.dumps({"data": {"index.html": "portal-version=v2\n"}})
        patch_v1 = json.dumps({"data": {"index.html": "portal-version=v1\n"}})
        try:
            kube.run(["patch", "configmap", "portal-content", "-n", kube.namespace, "--type=merge", "-p", patch_v2])
        except KubectlError as exc:
            _raise_candidate_mutation_failure(kube, exc, "the required portal ConfigMap update")
        try:
            kube.wait_until(
                lambda: _all_ready_pods_serve_base64(kube, "app=portal-web", "web", 2, "cG9ydGFsLXZlcnNpb249djIK"),
                "both Ready portal pods to serve portal-version=v2 without rollout",
                120,
                2,
            )
            after = {pod.get("metadata", {}).get("uid") for pod in kube.list("pods", "app=portal-web")}
            require(after == before, "portal ConfigMap update triggered or coincided with a pod rollout")
        finally:
            kube.run(["patch", "configmap", "portal-content", "-n", kube.namespace, "--type=merge", "-p", patch_v1])
            kube.wait_until(
                lambda: _all_ready_pods_serve_base64(kube, "app=portal-web", "web", 2, "cG9ydGFsLXZlcnNpb249djEK"),
                "both Ready portal pods to serve the restored v1 body",
                120,
                2,
            )
        return "portal-web wiring is exact and projected ConfigMap changes appeared and were restored without rollout"

    def r05() -> str:
        _assert_medium_hardening(kube.get("deployment", "portal-web"), "portal-web")
        return "portal-web has every required pod and container hardening field"

    def r06() -> str:
        deployment = kube.get("deployment", "portal-web")
        service = kube.get("service", "portal-svc")
        spec = service.get("spec", {})
        require(spec.get("type", "ClusterIP") == "ClusterIP", "portal-svc is not ClusterIP")
        require(spec.get("selector") == {"app": "portal-web"}, "portal-svc selector differs")
        port = _service_port(service, 80)
        require(port.get("protocol", "TCP") == "TCP" and port.get("targetPort") == "http", "portal-svc must target named TCP port http")
        require(_resolved_target_port(port, deployment) == "8080", "portal-svc targetPort does not resolve to 8080")
        cluster_ip = spec.get("clusterIP")
        require(cluster_ip not in {None, "", "None"}, "portal-svc has no ClusterIP")
        _assert_exact_body_probe(kube, "aipc-eval-portal-initial", f"http://{cluster_ip}:80/", "portal-version=v1", 18, labels={"access": "portal"})
        pods = [pod for pod in kube.list("pods", "app=portal-web") if _ready_pod(pod)]
        require(len(pods) == 2, f"expected two Ready portal pods, found {len(pods)}")
        victim = pods[0]
        victim_name = victim.get("metadata", {}).get("name", "")
        victim_uid = victim.get("metadata", {}).get("uid", "")
        kube.run(["delete", "pod", victim_name, "-n", kube.namespace, "--wait=true"])
        kube.wait_until(
            lambda: victim_uid not in _ready_endpoint_uids(kube.get("endpoints", "portal-svc")),
            "the deleted portal pod to leave the Ready endpoint set",
            30,
            1,
        )
        probe = kube.run_probe_pod(
            "aipc-eval-portal-continuity",
            f'i=0; while [ "$i" -lt 10 ]; do test "$(wget -q -T 5 -O - http://{cluster_ip}:80/)" = portal-version=v1; i=$((i + 1)); sleep 1; done',
            labels={"access": "portal"},
            timeout_seconds=30,
        )
        kube.wait_until(
            lambda: any(
                pod.get("metadata", {}).get("uid") not in {victim_uid, ""} and _ready_pod(pod)
                for pod in kube.list("pods", "app=portal-web")
            ),
            "a distinct Ready portal replacement pod",
            60,
            1,
        )
        return f"portal-svc returned the exact body through ten replacement-time requests; {probe}"

    def r07() -> str:
        _assert_single_ingress_policy(kube.get("networkpolicy", "portal-ingress"), app="portal-web", access="portal", port=8080)
        cluster_ip = kube.get("service", "portal-svc").get("spec", {}).get("clusterIP")
        allowed = kube.run_probe_pod("aipc-eval-portal-allowed", f"wget -q -T 5 -O /dev/null http://{cluster_ip}:80/", labels={"access": "portal"})
        denied = kube.run_probe_pod("aipc-eval-portal-denied", f"wget -q -T 4 -O /dev/null http://{cluster_ip}:80/", labels={"access": "other"}, expect_success=False)
        return f"portal ingress is structurally narrow and works only for the allowed label; {allowed}; {denied}"

    def r08() -> str:
        return _assert_exact_pdb(kube, "portal-pdb", "portal-web")

    def r09() -> str:
        spec = kube.get("deployment", "portal-web").get("spec", {})
        strategy = spec.get("strategy", {})
        require(strategy.get("type", "RollingUpdate") == "RollingUpdate", "portal-web strategy must be RollingUpdate")
        rolling = strategy.get("rollingUpdate", {})
        require(int_or_string(rolling.get("maxUnavailable")) == "0", "maxUnavailable must be 0")
        require(int_or_string(rolling.get("maxSurge")) == "1", "maxSurge must be 1")
        require(spec.get("minReadySeconds") == 5, "minReadySeconds must be 5")
        require(spec.get("progressDeadlineSeconds") == 90, "progressDeadlineSeconds must be 90")
        require(spec.get("revisionHistoryLimit") == 2, "revisionHistoryLimit must be 2")
        return "portal-web has the exact rollout controls"

    def r10() -> str:
        require(kube.get("secret", "portal-token").get("immutable") is True, "portal-token must be immutable")
        return "portal-token is immutable"

    return [
        ("R01", "isolated medium portal namespace", lambda: _check_namespace(kube, "medium-portal-ns", _require_candidate_path(candidate_path))),
        ("R02", "exact portal content", r02),
        ("R03", "exact portal token", r03),
        ("R04", "portal Deployment and live configuration projection", r04),
        ("R05", "portal pod hardening", r05),
        ("R06", "portal Service response and replacement continuity", r06),
        ("R07", "portal ingress isolation", r07),
        ("R08", "portal disruption budget", r08),
        ("R09", "portal rollout controls", r09),
        ("R10", "immutable portal token", r10),
    ]


def _medium2_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        _assert_exact_configmap_data(kube.get("configmap", "batch-input"), {"VALUE": "17", "FACTOR": "3"})
        return "batch-input contains exactly VALUE=17 and FACTOR=3"

    def r03() -> str:
        secret = kube.get("secret", "batch-mode")
        require(secret.get("type") == "Opaque", "batch-mode must be Opaque")
        require(set(secret.get("data", {})) == {"OPERATION"}, "batch-mode must contain only OPERATION")
        require(_decode_secret_value(secret.get("data", {}).get("OPERATION"), "OPERATION") == "multiply", "OPERATION differs")
        return "batch-mode is the exact one-key Opaque Secret"

    def r04() -> str:
        account = kube.get("serviceaccount", "batch-runner")
        require(account.get("automountServiceAccountToken") is True, "batch-runner automountServiceAccountToken must be true")
        return "batch-runner exists with token automount explicitly enabled"

    def r05() -> str:
        role = kube.get("role", "batch-input-reader")
        require(role.get("rules") == [{"apiGroups": [""], "resources": ["configmaps"], "resourceNames": ["batch-input"], "verbs": ["get"]}], "batch-input-reader rule differs")
        return "batch-input-reader grants only get on ConfigMap batch-input"

    def r06() -> str:
        binding = kube.get("rolebinding", "batch-input-reader-binding")
        require(binding.get("subjects") == [{"kind": "ServiceAccount", "name": "batch-runner", "namespace": "medium-batch-ns"}], "RoleBinding must bind only batch-runner")
        role_ref = binding.get("roleRef", {})
        require(role_ref == {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": "batch-input-reader"}, "RoleBinding roleRef differs")
        require(kube.auth_can_i("batch-runner", "get", "configmap/batch-input"), "batch-runner cannot initially get batch-input")
        require(not kube.auth_can_i("batch-runner", "list", "configmaps"), "batch-runner can list ConfigMaps")
        restore = {
            "apiVersion": binding.get("apiVersion", "rbac.authorization.k8s.io/v1"),
            "kind": "RoleBinding",
            "metadata": {"name": "batch-input-reader-binding", "namespace": kube.namespace},
            "roleRef": role_ref,
            "subjects": binding.get("subjects"),
        }
        kube.run(["delete", "rolebinding", "batch-input-reader-binding", "-n", kube.namespace, "--wait=true"])
        try:
            kube.wait_until(lambda: not kube.auth_can_i("batch-runner", "get", "configmap/batch-input"), "batch-runner get authorization to become denied", 15, 1)
        finally:
            kube.run(["apply", "-f", "-"], input_text=json.dumps(restore))
        kube.wait_until(lambda: kube.auth_can_i("batch-runner", "get", "configmap/batch-input"), "batch-runner get authorization to recover", 15, 1)
        return "batch-runner authorization follows the required allow-deny-allow transition"

    def r07() -> str:
        hard = kube.get("resourcequota", "batch-limits").get("spec", {}).get("hard", {})
        require(set(hard) == {"count/jobs.batch", "pods"}, "batch-limits has missing or additional hard limits")
        require({key: int_or_string(value) for key, value in hard.items()} == {"count/jobs.batch": "3", "pods": "6"}, "batch-limits values differ")
        return "batch-limits has exactly the requested Job and pod limits"

    def r08() -> str:
        job = kube.get("job", "multiply-job")
        spec = job.get("spec", {})
        pod = spec.get("template", {}).get("spec", {})
        require(pod.get("serviceAccountName", "default") == "batch-runner", "multiply-job does not use batch-runner")
        require(pod.get("restartPolicy") == "Never", "multiply-job restartPolicy must be Never")
        containers = pod.get("containers", [])
        require(len(containers) == 1 and containers[0].get("name") == "calculator", "multiply-job must have one calculator container")
        container = containers[0]
        require(container.get("image") == "busybox:1.36.1", "calculator image differs")
        require(_combined_command(container) == ["sh", "-c", 'test "$OPERATION" = multiply && result=$((VALUE * FACTOR)) && echo result=$result && test "$result" -eq 51'], "calculator command differs")
        env = {item.get("name"): item.get("valueFrom", {}) for item in container.get("env", []) if isinstance(item, dict)}
        require(env.get("VALUE", {}).get("configMapKeyRef", {}) == {"name": "batch-input", "key": "VALUE"}, "VALUE source differs")
        require(env.get("FACTOR", {}).get("configMapKeyRef", {}) == {"name": "batch-input", "key": "FACTOR"}, "FACTOR source differs")
        require(env.get("OPERATION", {}).get("secretKeyRef", {}) == {"name": "batch-mode", "key": "OPERATION"}, "OPERATION source differs")
        require(spec.get("backoffLimit") == 1, "backoffLimit must be 1")
        require(spec.get("activeDeadlineSeconds") == 60, "activeDeadlineSeconds must be 60")
        kube.wait_until(lambda: kube.get("job", "multiply-job").get("status", {}).get("succeeded", 0) == 1, "multiply-job to complete", 60, 1)
        pods = kube.list("pods", "job-name=multiply-job")
        require(len(pods) == 1, f"multiply-job must have exactly one pod, found {len(pods)}")
        logs = kube.run(["logs", pods[0].get("metadata", {}).get("name", ""), "-n", kube.namespace]).stdout
        require(logs == "result=51\n", f"multiply-job logs differ: {logs!r}")
        return "multiply-job has exact input wiring, completes once, and logs result=51"

    def r09() -> str:
        spec = kube.get("job", "multiply-job").get("spec", {})
        require(spec.get("completions") == 1, "completions must be 1")
        require(spec.get("parallelism") == 1, "parallelism must be 1")
        require(spec.get("completionMode") == "NonIndexed", "completionMode must be NonIndexed")
        require(spec.get("ttlSecondsAfterFinished") == 600, "ttlSecondsAfterFinished must be 600")
        return "multiply-job has the exact completion and lifecycle controls"

    def r10() -> str:
        require(kube.get("secret", "batch-mode").get("immutable") is True, "batch-mode must be immutable")
        return "batch-mode is immutable"

    return [
        ("R01", "isolated medium batch namespace", lambda: _check_namespace(kube, "medium-batch-ns", _require_candidate_path(candidate_path))),
        ("R02", "exact batch input", r02),
        ("R03", "exact batch mode", r03),
        ("R04", "batch service identity", r04),
        ("R05", "single-object batch Role", r05),
        ("R06", "RoleBinding authorization loss and recovery", r06),
        ("R07", "batch resource quota", r07),
        ("R08", "multiply Job completion and output", r08),
        ("R09", "multiply Job lifecycle", r09),
        ("R10", "immutable batch mode", r10),
    ]


def _medium3_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    def r02() -> str:
        _assert_exact_configmap_data(kube.get("configmap", "registry-prefix"), {"PREFIX": "registry"})
        return "registry-prefix contains exactly PREFIX=registry"

    def r03() -> str:
        secret = kube.get("secret", "registry-suffix")
        require(secret.get("type") == "Opaque", "registry-suffix must be Opaque")
        require(set(secret.get("data", {})) == {"SUFFIX"}, "registry-suffix must contain only SUFFIX")
        require(_decode_secret_value(secret.get("data", {}).get("SUFFIX"), "SUFFIX") == "stable", "SUFFIX differs")
        return "registry-suffix is the exact one-key Opaque Secret"

    def r04() -> str:
        service = kube.get("service", "registry-headless")
        spec = service.get("spec", {})
        require(spec.get("clusterIP") == "None", "registry-headless is not headless")
        require(spec.get("publishNotReadyAddresses", False) is False, "publishNotReadyAddresses must be false")
        require(spec.get("selector") == {"app": "registry"}, "registry-headless selector differs")
        port = _service_port(service, 8080)
        require(port.get("protocol", "TCP") == "TCP" and port.get("targetPort") == "http", "registry-headless port mapping differs")
        return "registry-headless has the exact headless selector and named-port mapping"

    def r05() -> str:
        stateful = kube.get("statefulset", "registry")
        spec = stateful.get("spec", {})
        require(spec.get("serviceName") == "registry-headless", "registry serviceName differs")
        require(spec.get("replicas") == 2, "registry must have two replicas")
        require(pod_labels(stateful).get("app") == "registry", "pod label app=registry is missing")
        container = _single_named_container(stateful, "registry")
        require(container.get("image") == "busybox:1.36.1", "registry image differs")
        expected_command = 'if [ ! -f /data/index.html ]; then printf "%s-%s-%s\\n" "$PREFIX" "$HOSTNAME" "$SUFFIX" > /data/index.html; fi; httpd -f -p 8080 -h /data'
        require(_combined_command(container) == ["sh", "-c", expected_command], "registry command differs")
        env = {item.get("name"): item.get("valueFrom", {}) for item in container.get("env", []) if isinstance(item, dict)}
        require(env.get("PREFIX", {}).get("configMapKeyRef", {}) == {"name": "registry-prefix", "key": "PREFIX"}, "PREFIX source differs")
        require(env.get("SUFFIX", {}).get("secretKeyRef", {}) == {"name": "registry-suffix", "key": "SUFFIX"}, "SUFFIX source differs")
        require(not any(item.get("secretRef", {}).get("name") == "registry-suffix" for item in container.get("envFrom", [])), "registry-suffix must not be exposed through envFrom")
        _assert_tcp_port(container, 8080, "http")
        _assert_http_readiness(container, path="/", port="http", period=5)
        mount = find_mount(container, "/data")
        require(mount is not None and mount.get("name") == "data", "data claim is not mounted at /data")
        assert_claim(volume_claim_template(stateful, "data"), "64Mi")
        kube.wait_until(lambda: kube.get("statefulset", "registry").get("status", {}).get("readyReplicas", 0) == 2, "registry to have two Ready replicas", 90, 1)
        original = kube.get("pod", "registry-0")
        original_uid = original.get("metadata", {}).get("uid", "")
        kube.exec("registry-0", "printf oracle-marker > /data/oracle-marker", "registry")
        kube.run(["delete", "pod", "registry-0", "-n", kube.namespace, "--wait=true"])
        replacement: dict[str, Any] = {}

        def recovered() -> bool:
            nonlocal replacement
            candidates = [
                pod
                for pod in kube.list("pods", "app=registry")
                if pod.get("metadata", {}).get("name") == "registry-0"
            ]
            if not candidates:
                return False
            candidate = candidates[0]
            if candidate.get("metadata", {}).get("uid") in {"", original_uid} or not _ready_pod(candidate):
                return False
            replacement = candidate
            return True

        kube.wait_until(recovered, "a distinct Ready registry-0 replacement", 90, 1)
        kube.exec("registry-0", 'test "$(cat /data/oracle-marker)" = oracle-marker', "registry")
        kube.wait_until(lambda: len(_ready_endpoint_uids(kube.get("endpoints", "registry-headless"))) == 2, "registry-headless to regain two Ready endpoints", 90, 1)
        kube.exec("registry-0", "rm -f /data/oracle-marker", "registry")
        return f"registry-0 retained its PVC marker after replacement {original_uid} -> {replacement.get('metadata', {}).get('uid')} and the headless Service regained two endpoints"

    def r06() -> str:
        _assert_medium_hardening(kube.get("statefulset", "registry"), "registry")
        return "registry has every required pod and container hardening field"

    def r07() -> str:
        _assert_single_ingress_policy(kube.get("networkpolicy", "registry-ingress"), app="registry", access="registry", port=8080)
        allowed = kube.run_probe_pod("aipc-eval-registry-allowed", "wget -q -T 5 -O /dev/null http://registry-headless:8080/", labels={"access": "registry"})
        denied = kube.run_probe_pod("aipc-eval-registry-denied", "wget -q -T 4 -O /dev/null http://registry-headless:8080/", labels={"access": "other"}, expect_success=False)
        return f"registry ingress is structurally narrow and live-isolated; {allowed}; {denied}"

    def r08() -> str:
        return _assert_exact_pdb(kube, "registry-pdb", "registry")

    def r09() -> str:
        retention = kube.get("statefulset", "registry").get("spec", {}).get("persistentVolumeClaimRetentionPolicy", {})
        require(retention == {"whenDeleted": "Retain", "whenScaled": "Retain"}, "PVC retention policy differs")
        return "registry retains PVCs when deleted and scaled"

    def r10() -> str:
        spec = kube.get("statefulset", "registry").get("spec", {})
        require(spec.get("podManagementPolicy") == "OrderedReady", "podManagementPolicy must be OrderedReady")
        update = spec.get("updateStrategy", {})
        require(update.get("type", "RollingUpdate") == "RollingUpdate", "updateStrategy must be RollingUpdate")
        require(update.get("rollingUpdate", {}).get("partition", 0) == 0, "RollingUpdate partition must be 0")
        return "registry uses OrderedReady and RollingUpdate partition 0"

    return [
        ("R01", "isolated medium stateful namespace", lambda: _check_namespace(kube, "medium-stateful-ns", _require_candidate_path(candidate_path))),
        ("R02", "exact registry prefix", r02),
        ("R03", "exact registry suffix", r03),
        ("R04", "registry headless Service", r04),
        ("R05", "registry StatefulSet storage and recovery", r05),
        ("R06", "registry pod hardening", r06),
        ("R07", "registry ingress isolation", r07),
        ("R08", "registry disruption budget", r08),
        ("R09", "registry PVC retention", r09),
        ("R10", "registry ordering and update strategy", r10),
    ]


def _medium4_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    expected_cron_command = 'i=0; until body=$(wget -qO- http://check-svc); do i=$((i + 1)); [ "$i" -ge 10 ] && exit 1; sleep 2; done; test "$body" = check-ready && echo check-ok'

    def r02() -> str:
        _assert_exact_configmap_data(kube.get("configmap", "check-content"), {"index.html": "check-ready\n"})
        return "check-content contains exactly the required body"

    def r03() -> str:
        deployment = kube.get("deployment", "check-web")
        require(deployment.get("spec", {}).get("replicas") == 2, "check-web must have two replicas")
        require(pod_labels(deployment).get("app") == "check-web", "pod label app=check-web is missing")
        container = _single_named_container(deployment, "web")
        require(container.get("image") == "busybox:1.36.1", "check-web image differs")
        require(_combined_command(container) in (["httpd", "-f", "-p", "8080", "-h", "/www"], ["sh", "-c", "httpd -f -p 8080 -h /www"]), "check-web command differs")
        mount = find_mount(container, "/www")
        require(mount is not None and mount.get("readOnly") is True, "/www must be mounted read-only")
        volume = find_volume(pod_spec(deployment), mount.get("name", ""))
        require(volume is not None and volume.get("configMap", {}).get("name") == "check-content", "/www is not backed by check-content")
        require(not volume.get("configMap", {}).get("items"), "check-content must be mounted completely")
        _assert_tcp_port(container, 8080, "http")
        _assert_http_readiness(container, path="/", port="http", period=5, failure=3, initial_delay=2)
        kube.rollout("deployment", "check-web", timeout_seconds=60)
        return "check-web has exact content wiring, listener, probe, and two Ready replicas"

    def r04() -> str:
        service = kube.get("service", "check-svc")
        spec = service.get("spec", {})
        require(spec.get("type", "ClusterIP") == "ClusterIP", "check-svc is not ClusterIP")
        require(spec.get("selector") == {"app": "check-web"}, "check-svc selector differs")
        port = _service_port(service, 80)
        require(port.get("protocol", "TCP") == "TCP" and port.get("targetPort") == "http", "check-svc port mapping differs")
        require(_resolved_target_port(port, kube.get("deployment", "check-web")) == "8080", "check-svc targetPort does not resolve to 8080")
        cluster_ip = spec.get("clusterIP")
        require(cluster_ip not in {None, "", "None"}, "check-svc has no ClusterIP")
        probe = _assert_exact_body_probe(kube, "aipc-eval-check-service", f"http://{cluster_ip}:80/", "check-ready", 12, labels={"access": "checker"})
        return f"check-svc returns the exact body to a permitted client; {probe}"

    def r05() -> str:
        _assert_medium_hardening(kube.get("deployment", "check-web"), "check-web")
        return "check-web has every required pod and container hardening field"

    def r06() -> str:
        _assert_single_ingress_policy(kube.get("networkpolicy", "check-ingress"), app="check-web", access="checker", port=8080)
        cluster_ip = kube.get("service", "check-svc").get("spec", {}).get("clusterIP")
        allowed = kube.run_probe_pod("aipc-eval-check-allowed", f"wget -q -T 5 -O /dev/null http://{cluster_ip}:80/", labels={"access": "checker"})
        denied = kube.run_probe_pod("aipc-eval-check-denied", f"wget -q -T 4 -O /dev/null http://{cluster_ip}:80/", labels={"access": "other"}, expect_success=False)
        return f"check ingress is structurally narrow and live-isolated; {allowed}; {denied}"

    def r07() -> str:
        cronjob = kube.get("cronjob", "scheduled-check")
        spec = cronjob.get("spec", {})
        require(spec.get("schedule") == "*/5 * * * *", "scheduled-check schedule differs")
        require(spec.get("suspend") is True, "scheduled-check must be suspended")
        require(spec.get("concurrencyPolicy") == "Forbid", "concurrencyPolicy must be Forbid")
        require(spec.get("successfulJobsHistoryLimit") == 1, "successfulJobsHistoryLimit must be 1")
        require(spec.get("failedJobsHistoryLimit") == 1, "failedJobsHistoryLimit must be 1")
        pod = _cron_spec(cronjob)
        require(cronjob.get("spec", {}).get("jobTemplate", {}).get("spec", {}).get("template", {}).get("metadata", {}).get("labels") == {"access": "checker"}, "CronJob pod label must be exactly access=checker")
        require(pod.get("restartPolicy") == "Never", "CronJob restartPolicy must be Never")
        containers = pod.get("containers", [])
        require(len(containers) == 1 and containers[0].get("name") == "checker", "CronJob must have one checker container")
        require(containers[0].get("image") == "busybox:1.36.1", "checker image differs")
        require(_combined_command(containers[0]) == ["sh", "-c", expected_cron_command], "scheduled-check command differs")
        return _run_cronjob_and_assert_log(kube, "scheduled-check", "scheduled-check", "check-ok\n", 60)

    def r08() -> str:
        return _assert_exact_pdb(kube, "check-pdb", "check-web")

    def r09() -> str:
        spec = kube.get("deployment", "check-web").get("spec", {})
        strategy = spec.get("strategy", {})
        require(strategy.get("type", "RollingUpdate") == "RollingUpdate", "check-web strategy must be RollingUpdate")
        rolling = strategy.get("rollingUpdate", {})
        require(int_or_string(rolling.get("maxUnavailable")) == "0", "maxUnavailable must be 0")
        require(int_or_string(rolling.get("maxSurge")) == "1", "maxSurge must be 1")
        require(spec.get("minReadySeconds") == 5, "minReadySeconds must be 5")
        require(spec.get("progressDeadlineSeconds") == 90, "progressDeadlineSeconds must be 90")
        require(spec.get("revisionHistoryLimit") == 2, "revisionHistoryLimit must be 2")
        return "check-web has the exact rollout controls"

    return [
        ("R01", "isolated medium checker namespace", lambda: _check_namespace(kube, "medium-check-ns", _require_candidate_path(candidate_path))),
        ("R02", "exact checker content", r02),
        ("R03", "checker Deployment and readiness", r03),
        ("R04", "exact checker Service response", r04),
        ("R05", "checker pod hardening", r05),
        ("R06", "checker ingress isolation", r06),
        ("R07", "suspended executable checker CronJob", r07),
        ("R08", "checker disruption budget", r08),
        ("R09", "checker rollout controls", r09),
    ]


def _medium5_checks(
    kube: Kubectl, candidate_path: Path | None = None
) -> list[tuple[str, str, Callable[[], str | None]]]:
    expected_cron_command = 'i=0; until body=$(wget -qO- http://ledger-svc); do i=$((i + 1)); [ "$i" -ge 10 ] && exit 1; sleep 2; done; test "$body" = ledger-version=v1 && echo ledger-check-ok'

    def r02() -> str:
        _assert_exact_configmap_data(kube.get("configmap", "ledger-seed"), {"index.html": "ledger-version=v1\n"})
        return "ledger-seed contains exactly the initial ledger body"

    def r03() -> str:
        claim = kube.get("persistentvolumeclaim", "ledger-data")
        spec = claim.get("spec", {})
        require(set(spec.get("accessModes", [])) == {"ReadWriteOnce"}, "ledger-data must be ReadWriteOnce")
        require(spec.get("volumeMode", "Filesystem") == "Filesystem", "ledger-data volumeMode must be Filesystem")
        require(spec.get("resources", {}).get("requests", {}).get("storage") == "64Mi", "ledger-data storage request must be 64Mi")
        return "ledger-data has the exact access mode, volume mode, and size"

    def r04() -> str:
        deployment = kube.get("deployment", "ledger-web")
        require(deployment.get("spec", {}).get("replicas") == 1, "ledger-web must have one replica")
        require(pod_labels(deployment).get("app") == "ledger-web", "pod label app=ledger-web is missing")
        pod = pod_spec(deployment)
        init = pod.get("initContainers", [])
        main = pod.get("containers", [])
        require(len(init) == 1 and init[0].get("name") == "seed", "ledger-web must have exactly one seed init container")
        require(len(main) == 1 and main[0].get("name") == "web", "ledger-web must have exactly one web container")
        require(init[0].get("image") == "busybox:1.36.1", "seed image differs")
        require(_combined_command(init[0]) == ["sh", "-c", "if [ ! -f /data/index.html ]; then cp /seed/index.html /data/index.html; fi"], "seed command differs")
        require(main[0].get("image") == "busybox:1.36.1", "web image differs")
        require(_combined_command(main[0]) in (["httpd", "-f", "-p", "8080", "-h", "/data"], ["sh", "-c", "httpd -f -p 8080 -h /data"]), "ledger web command differs")
        seed_mount = find_mount(init[0], "/seed")
        init_data_mount = find_mount(init[0], "/data")
        web_data_mount = find_mount(main[0], "/data")
        require(seed_mount is not None and seed_mount.get("readOnly") is True, "/seed must be mounted read-only in seed")
        require(init_data_mount is not None and web_data_mount is not None and init_data_mount.get("name") == web_data_mount.get("name"), "seed and web must share the /data volume")
        seed_volume = find_volume(pod, seed_mount.get("name", ""))
        data_volume = find_volume(pod, web_data_mount.get("name", ""))
        require(seed_volume is not None and seed_volume.get("configMap", {}).get("name") == "ledger-seed", "/seed is not backed by ledger-seed")
        require(data_volume is not None and data_volume.get("persistentVolumeClaim", {}).get("claimName") == "ledger-data", "/data is not backed by ledger-data")
        _assert_tcp_port(main[0], 8080, "http")
        _assert_http_readiness(main[0], path="/", port="http", period=5)
        kube.rollout("deployment", "ledger-web", timeout_seconds=90)
        pods = [pod for pod in kube.list("pods", "app=ledger-web") if _ready_pod(pod)]
        require(len(pods) == 1, f"expected one Ready ledger pod, found {len(pods)}")
        original_name = pods[0].get("metadata", {}).get("name", "")
        original_uid = pods[0].get("metadata", {}).get("uid", "")
        kube.exec(original_name, "printf 'ledger-version=v2\\n' > /data/index.html", "web")
        kube.exec(original_name, 'test "$(base64 /data/index.html | tr -d \'\\n\')" = bGVkZ2VyLXZlcnNpb249djIK', "web")
        kube.run(["delete", "pod", original_name, "-n", kube.namespace, "--wait=true"])
        replacement: dict[str, Any] = {}

        def recovered() -> bool:
            nonlocal replacement
            candidates = [pod for pod in kube.list("pods", "app=ledger-web") if _ready_pod(pod)]
            for candidate in candidates:
                if candidate.get("metadata", {}).get("uid") not in {"", original_uid}:
                    replacement = candidate
                    return True
            return False

        kube.wait_until(recovered, "a distinct Ready ledger replacement pod", 90, 1)
        replacement_name = replacement.get("metadata", {}).get("name", "")
        try:
            kube.exec(replacement_name, 'test "$(wget -qO- http://127.0.0.1:8080/ | base64 | tr -d \'\\n\')" = bGVkZ2VyLXZlcnNpb249djIK', "web")
        finally:
            kube.exec(replacement_name, "printf 'ledger-version=v1\\n' > /data/index.html", "web")
        kube.wait_until(lambda: _all_ready_pods_serve_base64(kube, "app=ledger-web", "web", 1, "bGVkZ2VyLXZlcnNpb249djEK"), "ledger service content to be restored to v1", 30, 1)
        return f"ledger PVC content survived replacement {original_uid} -> {replacement.get('metadata', {}).get('uid')} and was restored for later independent checks"

    def r05() -> str:
        service = kube.get("service", "ledger-svc")
        spec = service.get("spec", {})
        require(spec.get("type", "ClusterIP") == "ClusterIP", "ledger-svc is not ClusterIP")
        require(spec.get("selector") == {"app": "ledger-web"}, "ledger-svc selector differs")
        port = _service_port(service, 80)
        require(port.get("protocol", "TCP") == "TCP" and port.get("targetPort") == "http", "ledger-svc port mapping differs")
        require(_resolved_target_port(port, kube.get("deployment", "ledger-web")) == "8080", "ledger-svc targetPort does not resolve to 8080")
        cluster_ip = spec.get("clusterIP")
        require(cluster_ip not in {None, "", "None"}, "ledger-svc has no ClusterIP")
        probe = _assert_exact_body_probe(kube, "aipc-eval-ledger-service", f"http://{cluster_ip}:80/", "ledger-version=v1", 18, labels={"access": "ledger"})
        return f"ledger-svc returns the initial exact body to an allowed client; {probe}"

    def r06() -> str:
        _assert_single_ingress_policy(kube.get("networkpolicy", "ledger-ingress"), app="ledger-web", access="ledger", port=8080)
        cluster_ip = kube.get("service", "ledger-svc").get("spec", {}).get("clusterIP")
        allowed = kube.run_probe_pod("aipc-eval-ledger-allowed", f"wget -q -T 5 -O /dev/null http://{cluster_ip}:80/", labels={"access": "ledger"})
        denied = kube.run_probe_pod("aipc-eval-ledger-denied", f"wget -q -T 4 -O /dev/null http://{cluster_ip}:80/", labels={"access": "other"}, expect_success=False)
        return f"ledger ingress is structurally narrow and live-isolated; {allowed}; {denied}"

    def r07() -> str:
        hard = kube.get("resourcequota", "ledger-quota").get("spec", {}).get("hard", {})
        expected = {"requests.storage": "256Mi", "persistentvolumeclaims": "2", "pods": "6", "services": "2"}
        require(set(hard) == set(expected), "ledger-quota has missing or additional hard limits")
        require({key: int_or_string(value) for key, value in hard.items()} == expected, "ledger-quota values differ")
        return "ledger-quota has exactly the requested storage and object limits"

    def r08() -> str:
        cronjob = kube.get("cronjob", "ledger-check")
        spec = cronjob.get("spec", {})
        require(spec.get("schedule") == "*/15 * * * *", "ledger-check schedule differs")
        require(spec.get("suspend") is True, "ledger-check must be suspended")
        require(spec.get("concurrencyPolicy") == "Forbid", "ledger-check concurrencyPolicy must be Forbid")
        labels = spec.get("jobTemplate", {}).get("spec", {}).get("template", {}).get("metadata", {}).get("labels")
        require(labels == {"access": "ledger"}, "CronJob pod label must be exactly access=ledger")
        pod = _cron_spec(cronjob)
        require(pod.get("restartPolicy") == "Never", "ledger-check restartPolicy must be Never")
        containers = pod.get("containers", [])
        require(len(containers) == 1 and containers[0].get("name") == "checker", "ledger-check must have one checker container")
        require(containers[0].get("image") == "busybox:1.36.1", "ledger checker image differs")
        require(_combined_command(containers[0]) == ["sh", "-c", expected_cron_command], "ledger-check command differs")
        return _run_cronjob_and_assert_log(kube, "ledger-check", "ledger-check", "ledger-check-ok\n", 60)

    def r09() -> str:
        require(kube.get("configmap", "ledger-seed").get("immutable") is True, "ledger-seed must be immutable")
        return "ledger-seed is immutable"

    def r10() -> str:
        spec = kube.get("deployment", "ledger-web").get("spec", {})
        require(spec.get("strategy", {}).get("type") == "Recreate", "ledger-web strategy must be Recreate")
        require(spec.get("progressDeadlineSeconds") == 120, "progressDeadlineSeconds must be 120")
        require(spec.get("revisionHistoryLimit") == 1, "revisionHistoryLimit must be 1")
        return "ledger-web has the exact Recreate and history controls"

    return [
        ("R01", "isolated medium ledger namespace", lambda: _check_namespace(kube, "medium-ledger-ns", _require_candidate_path(candidate_path))),
        ("R02", "exact ledger seed", r02),
        ("R03", "ledger persistent claim", r03),
        # The public R08 contract says its CronJob is exercised before the
        # evaluator mutates storage, so initial-state checks deliberately run
        # before the destructive R04 replacement check.
        ("R05", "exact ledger Service response", r05),
        ("R06", "ledger ingress isolation", r06),
        ("R07", "ledger resource quota", r07),
        ("R08", "suspended live ledger checker", r08),
        ("R09", "immutable ledger seed", r09),
        ("R10", "ledger deployment strategy", r10),
        ("R04", "ledger Deployment and durable replacement", r04),
    ]


def _require_candidate_path(candidate_path: Path | None) -> Path:
    if candidate_path is None:
        raise EvaluationInfrastructureError(
            "the private evaluator was not given the deployed candidate manifest"
        )
    return candidate_path


def run_suite(
    task_id: str, kube: Kubectl, candidate_path: Path | None = None
) -> dict[str, Any]:
    if task_id == "pilot-001":
        checks = _task1_checks(kube, candidate_path)
    elif task_id == "pilot-002":
        checks = _task2_checks(kube, candidate_path)
    elif task_id == "pilot-003":
        checks = _task3_checks(kube, candidate_path)
    elif task_id == "pilot-004":
        checks = _task4_checks(kube, candidate_path)
    elif task_id == "easy-001":
        checks = _easy1_checks(kube, candidate_path)
    elif task_id == "easy-002":
        checks = _easy2_checks(kube, candidate_path)
    elif task_id == "easy-003":
        checks = _easy3_checks(kube, candidate_path)
    elif task_id == "easy-004":
        checks = _easy4_checks(kube, candidate_path)
    elif task_id == "easy-005":
        checks = _easy5_checks(kube, candidate_path)
    elif task_id == "medium-001":
        checks = _medium1_checks(kube, candidate_path)
    elif task_id == "medium-002":
        checks = _medium2_checks(kube, candidate_path)
    elif task_id == "medium-003":
        checks = _medium3_checks(kube, candidate_path)
    elif task_id == "medium-004":
        checks = _medium4_checks(kube, candidate_path)
    elif task_id == "medium-005":
        checks = _medium5_checks(kube, candidate_path)
    else:
        raise EvaluationError(f"unknown task id {task_id!r}")
    actual_ids = {requirement_id for requirement_id, _, _ in checks}
    require(actual_ids == EXPECTED_REQUIREMENTS[task_id], "suite requirement coverage is incomplete")
    collector = ResultCollector(task_id)
    for requirement_id, name, operation in checks:
        collector.check(requirement_id, name, operation)
    return collector.report()
