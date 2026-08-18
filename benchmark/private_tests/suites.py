from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path
from typing import Any, Callable

import yaml

from .core import (
    EvaluationError,
    EvaluationInfrastructureError,
    Kubectl,
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
    else:
        raise EvaluationError(f"unknown task id {task_id!r}")
    actual_ids = {requirement_id for requirement_id, _, _ in checks}
    require(actual_ids == EXPECTED_REQUIREMENTS[task_id], "suite requirement coverage is incomplete")
    collector = ResultCollector(task_id)
    for requirement_id, name, operation in checks:
        collector.check(requirement_id, name, operation)
    return collector.report()
