"""Hidden oracles for hard-006..hard-015.

The historical ``pending_`` filename is retained for import stability, but the
module is registered by ``suites.py``. The checks score public behavior: exact
values are asserted only where public wording is exact; shell implementations are
accepted when their live behavior and public bounds agree.  Evaluator mutations always
restore candidate resources in ``finally`` blocks.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Callable

from .core import (
    EvaluationInfrastructureError,
    Kubectl,
    KubectlError,
    RequirementFailure,
    all_containers,
    assert_claim,
    command_text,
    find_mount,
    find_volume,
    int_or_string,
    pod_labels,
    pod_spec,
    require,
    volume_claim_template,
)
from .very_hard_suites import (
    _all_pod_bodies,
    _binding,
    _body_probe,
    _claim,
    _container,
    _cron_job,
    _cron_spec,
    _deployment_controls,
    _endpoint_count,
    _endpoint_uids,
    _exact_configmap,
    _generations,
    _job_complete,
    _port,
    _ready,
    _ready_count,
    _replacement_continuity,
    _role,
    _scope,
    _snapshot,
    _stateful_controls,
    _uids,
)

Check = tuple[str, str, Callable[[], str | None]]

PENDING_EXPECTED_REQUIREMENTS = {
    task_id: {f"R{number:02d}" for number in range(1, count + 1)}
    for task_id, count in {
        "hard-006": 14,
        "hard-007": 14,
        "hard-008": 14,
        "hard-009": 13,
        "hard-010": 14,
        "hard-011": 13,
        "hard-012": 14,
        "hard-013": 14,
        "hard-014": 14,
        "hard-015": 14,
    }.items()
}


def _mutable_secret(kube: Kubectl, name: str, key: str, value: str) -> str:
    item = kube.get("secret", name)
    require(item.get("type") == "Opaque", f"{name} must be Opaque")
    require(item.get("immutable") is not True, f"{name} must remain mutable")
    require(set(item.get("data", {})) == {key}, f"{name} key set differs")
    try:
        actual = base64.b64decode(item["data"][key], validate=True).decode()
    except Exception as exc:
        raise RequirementFailure(f"{name}/{key} is not valid base64 UTF-8") from exc
    require(actual == value, f"{name}/{key} differs")
    return f"{name} is exact and mutable"


def _secret(kube: Kubectl, name: str, key: str, value: str) -> str:
    item = kube.get("secret", name)
    require(item.get("type") == "Opaque", f"{name} must be Opaque")
    require(item.get("immutable") is True, f"{name} must be immutable")
    require(set(item.get("data", {})) == {key}, f"{name} key set differs")
    try:
        decoded = base64.b64decode(item["data"][key], validate=True).decode("utf-8")
    except Exception as exc:
        raise RequirementFailure(f"{name}/{key} is not valid base64 UTF-8") from exc
    require(decoded == value, f"{name}/{key} differs")
    return f"{name} is exact and immutable"


def _token_policy(workload: dict[str, Any], expected: bool) -> None:
    require(
        pod_spec(workload).get("automountServiceAccountToken") is expected,
        "automountServiceAccountToken differs",
    )


def _source_name(volume: dict[str, Any], source_kind: str) -> str | None:
    source = volume.get(source_kind, {})
    if source_kind == "secret":
        name = source.get("secretName")
    elif source_kind == "persistentVolumeClaim":
        name = source.get("claimName")
    else:
        name = source.get("name")
    if name:
        return str(name)
    for item in volume.get("projected", {}).get("sources", []) or []:
        projected = item.get(source_kind, {}) if isinstance(item, dict) else {}
        projected_name = projected.get("name")
        if projected_name:
            return str(projected_name)
    return None


def _mount_source(
    workload: dict[str, Any],
    container: dict[str, Any],
    path: str | None,
    source_kind: str,
    source_name: str,
    *,
    read_only: bool | None = None,
) -> None:
    mount = find_mount(container, path)
    require(mount is not None, f"mount {path} is absent")
    volume = find_volume(pod_spec(workload), mount.get("name", ""))
    require(volume is not None, f"volume for {path} is absent")
    require(_source_name(volume, source_kind) == source_name, f"source for {path} differs")
    if read_only is False:
        require(mount.get("readOnly", False) is False, f"mount {path} must be writable")
    elif read_only is True and source_kind == "persistentVolumeClaim":
        require(mount.get("readOnly") is True, f"claim mount {path} must be read-only")
    # ConfigMap and Secret volumes are intrinsically read-only even when the
    # volumeMount omits the redundant readOnly flag.


def _secret_env(container: dict[str, Any], env: str, secret: str, key: str) -> None:
    matches = [
        item
        for item in container.get("env", []) or []
        if isinstance(item, dict) and item.get("name") == env
    ]
    require(len(matches) == 1, f"environment variable {env} must be declared once")
    reference = matches[0].get("valueFrom", {}).get("secretKeyRef", {})
    require(
        reference.get("name") == secret and reference.get("key") == key,
        f"{env} secretKeyRef differs",
    )
    require(reference.get("optional", False) is False, f"{env} must not make the Secret optional")


def _secret_input(
    workload: dict[str, Any],
    container: dict[str, Any],
    env: str,
    secret: str,
    key: str,
) -> None:
    """Accept secretKeyRef or a mounted Secret key when the task permits either."""

    try:
        _secret_env(container, env, secret, key)
        return
    except RequirementFailure:
        pass

    program = command_text(container)
    for mount in container.get("volumeMounts", []) or []:
        volume = find_volume(pod_spec(workload), mount.get("name", ""))
        if volume is None or _source_name(volume, "secret") != secret:
            continue
        source = volume.get("secret", {})
        projected_path = key
        items = source.get("items", []) or []
        if items:
            matches = [item for item in items if item.get("key") == key]
            if len(matches) != 1:
                continue
            projected_path = str(matches[0].get("path") or key)
        mount_path = str(mount.get("mountPath") or "").rstrip("/")
        candidate_paths = {mount_path}
        if mount.get("subPath") != key:
            candidate_paths.add(f"{mount_path}/{projected_path}")
        if any(path and path in program for path in candidate_paths):
            return
    raise RequirementFailure(f"{env} is not sourced from Secret {secret}/{key}")


def _probe(
    container: dict[str, Any],
    kind: str,
    path: str,
    port: str,
    period: int | None,
    failure: int | None,
) -> None:
    """Check probe semantics without prescribing named versus numeric ports."""

    probe = container.get(kind, {})
    valid_ports = {str(port)}
    for declared in container.get("ports", []) or []:
        if declared.get("name") == port and declared.get("containerPort") is not None:
            valid_ports.add(str(declared["containerPort"]))
    target = probe.get("httpGet")
    if isinstance(target, dict) and target:
        if path is not None:
            require(target.get("path") == path, f"{kind} HTTP path differs")
        require(int_or_string(target.get("port")) in valid_ports, f"{kind} HTTP port differs")
        require(target.get("scheme", "HTTP") == "HTTP", f"{kind} HTTP scheme differs")
    else:
        command = " ".join(str(item) for item in probe.get("exec", {}).get("command", []))
        require(
            re.search(r"\b(?:wget|curl)\b", command) is not None,
            f"{kind} does not make an HTTP request",
        )
        require(
            any(f":{candidate}" in command for candidate in valid_ports),
            f"{kind} HTTP port differs",
        )
        if path is not None:
            require(
                re.search(
                    rf"https?://[^\s'\"]+{re.escape(path)}(?:[\s'\"\);|&]|$)",
                    command,
                )
                is not None,
                f"{kind} HTTP path differs",
            )
    if period is not None:
        require(probe.get("periodSeconds", 10) == period, f"{kind} period differs")
    if failure is not None:
        require(probe.get("failureThreshold", 3) == failure, f"{kind} failure threshold differs")


def _hardened(workload: dict[str, Any], label: str) -> None:
    """Accept pod- or container-level UID/GID inheritance, but no weaker posture."""

    spec = pod_spec(workload)
    pod_security = spec.get("securityContext", {})
    require(pod_security.get("fsGroup") == 1000, f"{label} fsGroup differs")
    for container in all_containers(spec):
        context = container.get("securityContext", {})
        require(
            context.get("runAsUser", pod_security.get("runAsUser")) == 1000,
            f"{label} UID differs",
        )
        require(
            context.get("runAsGroup", pod_security.get("runAsGroup")) == 1000,
            f"{label} GID differs",
        )
        require(
            context.get("runAsNonRoot", pod_security.get("runAsNonRoot")) is not False,
            f"{label} explicitly permits root",
        )
        require(
            context.get("allowPrivilegeEscalation") is False,
            f"{label} permits privilege escalation",
        )
        require(
            context.get("readOnlyRootFilesystem") is True,
            f"{label} root filesystem is writable",
        )
        capabilities = context.get("capabilities", {})
        require("ALL" in set(capabilities.get("drop", []) or []), f"{label} does not drop ALL capabilities")
        require(not capabilities.get("add"), f"{label} adds capabilities after dropping ALL")


def _service(
    kube: Kubectl,
    name: str,
    selector: dict[str, str],
    port: int,
    target: str,
    *,
    headless: bool = False,
) -> dict[str, Any]:
    """Accept equivalent selector supersets and named or resolved numeric targets."""

    item = kube.get("service", name)
    spec = item.get("spec", {})
    require(spec.get("type", "ClusterIP") == "ClusterIP", f"{name} is not ClusterIP")
    actual_selector = spec.get("selector", {})
    for key, value in selector.items():
        require(actual_selector.get(key) == value, f"{name} selector misses {key}={value}")

    selector_text = ",".join(f"{key}={value}" for key, value in selector.items())
    matching_pods = kube.list("pods", selector_text)
    for pod in matching_pods:
        labels = pod.get("metadata", {}).get("labels", {})
        require(
            all(labels.get(key) == value for key, value in actual_selector.items()),
            f"{name} selector excludes a required pod",
        )

    if headless:
        require(spec.get("clusterIP") == "None", f"{name} is not headless")
        require(
            spec.get("publishNotReadyAddresses", False) is False,
            f"{name} publishes NotReady addresses",
        )
    else:
        require(spec.get("clusterIP") not in {None, "", "None"}, f"{name} lacks a cluster IP")

    ports = [candidate for candidate in spec.get("ports", []) or [] if candidate.get("port") == port]
    require(len(ports) == 1, f"{name} must expose the required port once")
    exposed = ports[0]
    require(exposed.get("protocol", "TCP") == "TCP", f"{name} port is not TCP")

    valid_targets = {str(target)}
    for pod in matching_pods:
        for container in pod.get("spec", {}).get("containers", []) or []:
            for declared in container.get("ports", []) or []:
                if declared.get("name") == target and declared.get("containerPort") is not None:
                    valid_targets.add(str(declared["containerPort"]))
    require(
        int_or_string(exposed.get("targetPort", port)) in valid_targets,
        f"{name} targetPort differs",
    )
    return item


def _pdb(kube: Kubectl, name: str, selector: dict[str, str], minimum: int) -> None:
    """Allow harmless selector labels only when they still select every required pod."""

    spec = kube.get("pdb", name).get("spec", {})
    actual = spec.get("selector", {}).get("matchLabels", {})
    for key, value in selector.items():
        require(actual.get(key) == value, f"{name} selector misses {key}={value}")
    selector_text = ",".join(f"{key}={value}" for key, value in selector.items())
    for pod in kube.list("pods", selector_text):
        labels = pod.get("metadata", {}).get("labels", {})
        require(
            all(labels.get(key) == value for key, value in actual.items()),
            f"{name} selector excludes a required pod",
        )
    require(int_or_string(spec.get("minAvailable")) == str(minimum), f"{name} minAvailable differs")
    require("maxUnavailable" not in spec, f"{name} must not set maxUnavailable")


def _limits(kube: Kubectl, limit_name: str, quota_name: str, expected: dict[str, str]) -> str:
    """Require the declared container defaults without rejecting unrelated range types."""

    entries = kube.get("limitrange", limit_name).get("spec", {}).get("limits", []) or []
    expected_request = {"cpu": "20m", "memory": "32Mi"}
    expected_limit = {"cpu": "200m", "memory": "128Mi"}
    matching = [
        entry
        for entry in entries
        if entry.get("type") == "Container"
        and entry.get("defaultRequest") == expected_request
        and entry.get("default") == expected_limit
    ]
    require(len(matching) == 1, f"{limit_name} container defaults differ")
    for entry in entries:
        if entry.get("type") != "Container" or entry is matching[0]:
            continue
        require(
            not entry.get("defaultRequest") and not entry.get("default"),
            f"{limit_name} has conflicting container defaults",
        )

    hard = kube.get("resourcequota", quota_name).get("spec", {}).get("hard", {})
    aliases = {
        "count/pods": "pods",
        "count/services": "services",
        "count/persistentvolumeclaims": "persistentvolumeclaims",
    }
    actual: dict[str, str] = {}
    for key, value in hard.items():
        normalized = aliases.get(key, key)
        rendered = int_or_string(value)
        require(
            normalized not in actual or actual[normalized] == rendered,
            f"{quota_name} declares conflicting quota aliases for {normalized}",
        )
        actual[normalized] = rendered
    require(actual == expected, f"{quota_name} hard limits differ")
    return "the declared container defaults and exact ResourceQuota are present"


def _candidate_mutation_failure(kube: Kubectl, exc: KubectlError, description: str) -> None:
    """Keep deterministic candidate rejection separate from evaluator transport failure."""

    detail = f"{exc.stdout}\n{exc.stderr}".lower()
    markers = ("immutable", "forbidden", "denied", "invalid", "not found", "notfound")
    if not any(marker in detail for marker in markers):
        raise exc
    ready = kube.run(
        ["get", "--raw=/readyz"],
        check=False,
        timeout=min(kube.command_timeout, 10),
    )
    if ready.returncode == 0 and "ok" in ready.stdout.lower():
        raise RequirementFailure(
            f"the evaluator could not perform {description} because the candidate rejected it while the API remained healthy"
        ) from exc
    raise exc


def _patch_data_value(kube: Kubectl, kind: str, name: str, key: str, value: str) -> None:
    stored = base64.b64encode(value.encode()).decode() if kind == "secret" else value
    try:
        kube.run([
            "patch",
            kind,
            name,
            "-n",
            kube.namespace,
            "--type=merge",
            "-p",
            json.dumps({"data": {key: stored}}),
        ])
    except KubectlError as exc:
        _candidate_mutation_failure(kube, exc, f"{kind}/{name} update")


def _data_value(kube: Kubectl, kind: str, name: str, key: str) -> str:
    value = kube.get(kind, name).get("data", {}).get(key)
    if kind == "configmap":
        return str(value)
    try:
        return base64.b64decode(value, validate=True).decode()
    except Exception as exc:
        raise RequirementFailure(f"{name}/{key} is not valid base64 UTF-8") from exc


def _restore_data_value(kube: Kubectl, kind: str, name: str, key: str, original: str) -> None:
    if _data_value(kube, kind, name, key) == original:
        return
    try:
        _patch_data_value(kube, kind, name, key, original)
    except Exception as exc:
        raise EvaluationInfrastructureError(f"could not restore {kind}/{name}") from exc


def _update_configmap(
    kube: Kubectl,
    name: str,
    key: str,
    changed: str,
    original: str,
    assertions: Callable[[str], bool],
    timeout: int,
    controllers: list[tuple[str, str]],
    selectors: list[str],
) -> str:
    generations = _generations(kube, controllers)
    uids = {selector: _uids(kube, selector) for selector in selectors}
    changed_applied = False
    try:
        _patch_data_value(kube, "configmap", name, key, changed)
        changed_applied = True
        kube.wait_until(lambda: assertions(changed), f"{name} live update", timeout, 2)
        require(_generations(kube, controllers) == generations, "controller generation changed")
        require({selector: _uids(kube, selector) for selector in selectors} == uids, "pod UID changed")
    finally:
        if changed_applied:
            _restore_data_value(kube, "configmap", name, key, original)
            kube.wait_until(lambda: assertions(original), f"{name} restoration", timeout, 2)
    if changed_applied:
        require(_generations(kube, controllers) == generations, "generation changed during restoration")
        require({selector: _uids(kube, selector) for selector in selectors} == uids, "UID changed during restoration")
    return f"{name} propagated in place and was restored without rollout"


def _workload_image_and_count(
    kube: Kubectl,
    kind: str,
    name: str,
    selector: str,
    replicas: int,
) -> dict[str, Any]:
    item = kube.get(kind, name)
    if kind != "daemonset":
        require(item.get("spec", {}).get("replicas") == replicas, f"{name} replicas differ")
    require(_ready_count(kube, selector) == replicas, f"{name} Ready count differs")
    # An unspecified init-container image is a valid implementation choice.
    for container in pod_spec(item).get("containers", []):
        require(container.get("image") == "busybox:1.36.1", f"{name} image differs")
    return item


def _emptydir_mount(workload: dict[str, Any], container: dict[str, Any], path: str) -> None:
    mount = find_mount(container, path)
    require(mount is not None, f"{path} mount is missing")
    volume = find_volume(pod_spec(workload), mount.get("name", ""))
    require(volume is not None and "emptyDir" in volume, f"{path} is not an emptyDir")


def _has_source_mount(workload: dict[str, Any], container: dict[str, Any], source_kind: str, source_name: str) -> None:
    matches = []
    for mount in container.get("volumeMounts", []):
        volume = find_volume(pod_spec(workload), mount.get("name", ""))
        if volume is None:
            continue
        source = volume.get(source_kind, {})
        projected = [
            item.get(source_kind, {})
            for item in volume.get("projected", {}).get("sources", [])
            if source_kind in item
        ]
        if source.get("name", source.get("claimName")) == source_name or any(
            item.get("name", item.get("claimName")) == source_name for item in projected
        ):
            matches.append(mount)
    require(len(matches) == 1, f"expected one mount sourced from {source_kind} {source_name}")


def _sole_container(workload: dict[str, Any], *, init: bool = False) -> dict[str, Any]:
    key = "initContainers" if init else "containers"
    items = pod_spec(workload).get(key, [])
    require(len(items) == 1, f"expected exactly one {key[:-1]}")
    return items[0]


def _quota_only(kube: Kubectl, name: str, expected: dict[str, str]) -> str:
    hard = kube.get("resourcequota", name).get("spec", {}).get("hard", {})
    actual = {key: int_or_string(value) for key, value in hard.items()}
    require(actual == expected, f"{name} hard limits differ")
    return f"{name} has the exact hard limits"


def _pdb_set(kube: Kubectl, entries: list[tuple[str, str, int]]) -> str:
    for name, app, minimum in entries:
        _pdb(kube, name, {"app": app}, minimum)
    return "all required PDBs have exact selectors and availability values"


def _pdbs_by_app(kube: Kubectl, expected: dict[str, int]) -> str:
    matches: dict[str, list[dict[str, Any]]] = {app: [] for app in expected}
    for item in kube.list("poddisruptionbudgets.policy"):
        spec = item.get("spec", {})
        labels = spec.get("selector", {}).get("matchLabels", {})
        if labels.get("app") in matches:
            matches[labels["app"]].append(item)
    for app, minimum in expected.items():
        require(len(matches[app]) == 1, f"expected one PDB selecting app={app}")
        spec = matches[app][0].get("spec", {})
        selector = spec.get("selector", {}).get("matchLabels", {})
        for pod in kube.list("pods", f"app={app}"):
            labels = pod.get("metadata", {}).get("labels", {})
            require(
                all(labels.get(key) == value for key, value in selector.items()),
                f"app={app} PDB selector excludes a required pod",
            )
        require(int_or_string(spec.get("minAvailable")) == str(minimum), f"app={app} PDB minAvailable differs")
        require("maxUnavailable" not in spec, f"app={app} PDB must not set maxUnavailable")
    return "one exact availability budget exists for every declared app"


def _labels_include(workload: dict[str, Any], expected: dict[str, str]) -> None:
    actual = pod_labels(workload)
    require(all(actual.get(key) == value for key, value in expected.items()), "required pod labels differ")


def _rolling_controls(workload: dict[str, Any], replicas: int, min_ready: int, deadline: int | None = None) -> None:
    spec = workload.get("spec", {})
    require(spec.get("replicas") == replicas, "replica count differs")
    rolling = spec.get("strategy", {}).get("rollingUpdate", {})
    require(int_or_string(rolling.get("maxUnavailable")) == "0", "maxUnavailable must be 0")
    require(int_or_string(rolling.get("maxSurge")) == "1", "maxSurge must be 1")
    require(spec.get("minReadySeconds") == min_ready, "minReadySeconds differs")
    if deadline is not None:
        require(spec.get("progressDeadlineSeconds") == deadline, "progressDeadlineSeconds differs")


def _exact_role_access(kube: Kubectl, name: str, expected: set[tuple[str, str, str, str]]) -> str:
    role = kube.get("role", name)
    require(not role.get("aggregationRule") and not role.get("nonResourceURLs"), f"{name} has unexpected access")
    actual: set[tuple[str, str, str, str]] = set()
    for rule in role.get("rules", []):
        require(not rule.get("nonResourceURLs"), f"{name} has non-resource access")
        for group in rule.get("apiGroups", []):
            for resource in rule.get("resources", []):
                for resource_name in rule.get("resourceNames", []):
                    for verb in rule.get("verbs", []):
                        actual.add((group, resource, resource_name, verb))
    require(actual == expected, f"{name} permissions differ")
    return f"{name} grants exactly the declared access"


def _service_outage(
    kube: Kubectl,
    service: str,
    down: Callable[[], bool],
    recovered: Callable[[], bool],
    down_timeout: int,
    recovery_timeout: int,
) -> str:
    saved = _snapshot(kube, "service", service)
    try:
        kube.run(["delete", "service", service, "-n", kube.namespace, "--wait=true"])
        kube.wait_until(down, f"{service} dependent readiness loss", down_timeout, 2)
    finally:
        try:
            kube.run(["apply", "-f", "-"], input_text=json.dumps(saved))
        except KubectlError as exc:
            raise EvaluationInfrastructureError(f"could not restore {service}") from exc
        kube.wait_until(recovered, f"{service} dependent recovery", recovery_timeout, 2)
    return f"{service} outage and restoration produced the required readiness transition"


def _secret_projection(
    kube: Kubectl,
    name: str,
    key: str,
    changed: str,
    original: str,
    assertions: Callable[[str], bool],
    controllers: list[tuple[str, str]],
    selectors: list[str],
    timeout: int,
) -> str:
    generations = _generations(kube, controllers)
    uids = {selector: _uids(kube, selector) for selector in selectors}
    changed_applied = False
    try:
        _patch_data_value(kube, "secret", name, key, changed)
        changed_applied = True
        kube.wait_until(lambda: assertions(changed), f"{name} projection", timeout, 2)
        require(_generations(kube, controllers) == generations, "controller generation changed")
        require({s: _uids(kube, s) for s in selectors} == uids, "pod UID changed")
    finally:
        if changed_applied:
            _restore_data_value(kube, "secret", name, key, original)
            kube.wait_until(lambda: assertions(original), f"{name} restoration", timeout, 2)
    if changed_applied:
        require(_generations(kube, controllers) == generations, "generation changed during restoration")
        require({s: _uids(kube, s) for s in selectors} == uids, "UID changed during restoration")
    return f"{name} projected and restored without rollout"


def _writable_data_paths(workload: dict[str, Any]) -> set[str]:
    spec = pod_spec(workload)
    claim_names = {
        item.get("metadata", {}).get("name")
        for item in workload.get("spec", {}).get("volumeClaimTemplates", [])
    }
    paths: set[str] = set()
    for container in all_containers(spec):
        for mount in container.get("volumeMounts", []):
            if mount.get("readOnly") is True:
                continue
            volume = find_volume(spec, mount.get("name", ""))
            data_backed = mount.get("name") in claim_names or (
                volume is not None
                and any(key in volume for key in ("emptyDir", "persistentVolumeClaim", "hostPath"))
            )
            if data_backed:
                paths.add(str(mount.get("mountPath", "")))
    return paths


def _harden_with_writes(kube: Kubectl, entries: list[tuple[str, str, set[str]]]) -> str:
    for kind, name, allowed in entries:
        item = kube.get(kind, name)
        if kind == "cronjob":
            item = {"spec": {"template": item["spec"]["jobTemplate"]["spec"]["template"]}}
        _hardened(item, name)
        require(_writable_data_paths(item) == allowed, f"{name} writable data paths differ")
    return "all containers are hardened with only the declared writable data paths"


def _local_body(kube: Kubectl, pod: str, container: str, port: int, expected: str) -> bool:
    encoded = base64.b64encode(expected.encode()).decode()
    result = kube.run(
        ["exec", "-n", kube.namespace, pod, "-c", container, "--", "sh", "-ec",
         f'test "$(wget -q -T 5 -O - http://127.0.0.1:{port}/ | base64 | tr -d \'\\n\')" = {encoded}'],
        check=False,
    )
    return result.returncode == 0


def _live_ingress_matrix(
    kube: Kubectl,
    allowed: list[tuple[str, str, dict[str, str]]],
    denied: list[tuple[str, str]],
) -> str:
    # Live outcomes avoid prescribing how equivalent NetworkPolicies are split or named.
    require(kube.list("networkpolicies"), "no NetworkPolicy objects were created")
    for probe, url, labels in allowed:
        kube.run_probe_pod(probe, f"wget -q -T 5 -O /dev/null {url}", labels=labels)
    for probe, url in denied:
        kube.run_probe_pod(probe, f"wget -q -T 4 -O /dev/null {url}", labels={}, expect_success=False, timeout_seconds=20)
    return "the required positive and negative NetworkPolicy paths were observed live"


def _replacement_with_stable_peer(
    kube: Kubectl,
    selector: str,
    service: str,
    expected: str,
    stable_selector: str,
    stable_service: str,
    stable_expected: str,
    access: dict[str, str],
    timeout: int,
) -> str:
    pods = [pod for pod in kube.list("pods", selector) if _ready(pod)]
    require(len(pods) == 2, "continuity test requires two Ready target replicas")
    stable_uids = _uids(kube, stable_selector)
    require(len(stable_uids) == 2, "stable peer does not begin with two Ready replicas")
    victim = pods[0]; old_uid = victim["metadata"]["uid"]
    kube.run(["delete", "pod", victim["metadata"]["name"], "-n", kube.namespace, "--wait=true"])
    kube.wait_until(lambda: old_uid not in _endpoint_uids(kube, service), "deleted endpoint removal", 30, 1)
    first = base64.b64encode(expected.encode()).decode()
    second = base64.b64encode(stable_expected.encode()).decode()
    command = (
        "i=0; while [ \"$i\" -lt 12 ]; do "
        f"test \"$(wget -q -T 5 -O - http://{service} | base64 | tr -d '\\n')\" = {first}; "
        f"test \"$(wget -q -T 5 -O - http://{stable_service} | base64 | tr -d '\\n')\" = {second}; "
        "i=$((i+1)); sleep 1; done"
    )
    kube.run_probe_pod(f"aipc-eval-{service}-peer"[:63], command, labels=access, timeout_seconds=45)
    kube.wait_until(lambda: any(_ready(p) and p["metadata"]["uid"] not in {old_uid,""} for p in kube.list("pods",selector)), "different-UID replacement", timeout, 1)
    require(_endpoint_count(kube, stable_service) == 2 and _uids(kube, stable_selector) == stable_uids, "stable peer changed during replacement")
    return "all twelve target and peer requests succeeded while only the target pod was replaced"


def _immutable_configmap(kube: Kubectl, name: str, expected: dict[str, str]) -> str:
    item = kube.get("configmap", name)
    require(item.get("data") == expected, f"{name} data differs")
    require(not item.get("binaryData"), f"{name} has unexpected binaryData")
    require(item.get("immutable") is True, f"{name} must be immutable")
    return f"{name} has exact immutable data"


def _as_job_workload(job: dict[str, Any]) -> dict[str, Any]:
    return {"spec": {"template": job["spec"]["template"]}}


def _one_job(kube: Kubectl, name: str, deadline: int, expected_log: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    job = kube.get("job", name)
    spec = job.get("spec", {})
    require(spec.get("backoffLimit") == 0, f"{name} backoffLimit differs")
    require(spec.get("activeDeadlineSeconds") == deadline, f"{name} deadline differs")
    require(spec.get("template", {}).get("spec", {}).get("restartPolicy") == "Never", f"{name} restartPolicy differs")
    workload = _as_job_workload(job)
    container = _sole_container(workload)
    require(container.get("image") == "busybox:1.36.1", f"{name} image differs")
    return workload, container, _job_complete(kube, name, deadline, expected_log)


def _one_cron_container(item: dict[str, Any]) -> dict[str, Any]:
    return _sole_container({"spec": {"template": item["spec"]["jobTemplate"]["spec"]["template"]}})


def _bounded_fetch(text: str, seconds: int | None = 5) -> None:
    """Accept common BusyBox timeout spellings instead of one reference command."""

    normalized = " ".join(text.lower().split())
    value = str(seconds) if seconds is not None else r"[1-9][0-9]*"
    patterns = (
        rf"(?:wget|curl)[^;\n]*(?:--timeout(?:=|\s+)|-t\s+|-t)(?:{value})(?:\s|$)",
        rf"\btimeout\s+(?:{value})(?:s)?\s+(?:wget|curl)\b",
        rf"(?:wget|curl)[^;\n]*--connect-timeout(?:=|\s+)(?:{value})(?:\s|$)",
        rf"curl[^;\n]*--max-time(?:=|\s+)(?:{value})(?:\s|$)",
    )
    label = f"a {seconds}-second-bounded" if seconds is not None else "a bounded"
    require(any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns), f"{label} request is absent")


def _bounded_attempts(text: str, maximum: int, sleep: int = 2) -> None:
    """Accept any positive attempt count up to the public maximum."""

    normalized = " ".join(text.split())
    require(re.search(r"\b(?:exit\s+[1-9][0-9]*|false)\b", normalized) is not None, "retry exhaustion does not fail non-zero")
    require(re.search(rf"\bsleep\s+{sleep}(?:\s|;|$)", normalized) is not None, f"retry interval must be {sleep} seconds")
    counts: list[int] = []
    for match in re.finditer(r"\bseq\s+(?:(0|1)\s+)?([0-9]+)\b", normalized):
        start, end = match.groups(); value = int(end)
        counts.append(value + 1 if start == "0" else value)
    for match in re.finditer(r"(?:-ge|-lt)\s+([0-9]+)\b", normalized):
        counts.append(int(match.group(1)))
    for match in re.finditer(r"-le\s+([0-9]+)\b", normalized):
        counts.append(int(match.group(1)))
    for match in re.finditer(r"-gt\s+([0-9]+)\b", normalized):
        counts.append(int(match.group(1)) + 1)
    require(any(1 <= value <= maximum for value in counts), f"no attempt bound at or below {maximum} was found")


def _recreate(workload: dict[str, Any]) -> None:
    require(workload.get("spec", {}).get("strategy", {}).get("type") == "Recreate", "Deployment strategy must be Recreate")


def _cron_structure(
    kube: Kubectl,
    name: str,
    schedule: str,
    deadline: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = kube.get("cronjob", name)
    spec = item.get("spec", {})
    require(spec.get("suspend") is True and spec.get("schedule") == schedule, f"{name} schedule/suspension differs")
    require(spec.get("concurrencyPolicy") == "Forbid", f"{name} concurrencyPolicy differs")
    require(spec.get("successfulJobsHistoryLimit") == 1 and spec.get("failedJobsHistoryLimit") == 1, f"{name} history limits differ")
    job_spec = spec.get("jobTemplate", {}).get("spec", {})
    pod = job_spec.get("template", {}).get("spec", {})
    require(
        job_spec.get("activeDeadlineSeconds") == deadline
        or pod.get("activeDeadlineSeconds") == deadline,
        f"{name} deadline differs",
    )
    require(pod.get("restartPolicy") == "Never", f"{name} restartPolicy differs")
    return item, _one_cron_container(item)


def _cron_contract(
    kube: Kubectl,
    name: str,
    schedule: str,
    deadline: int,
    job_name: str,
    log: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    item, container = _cron_structure(kube, name, schedule, deadline)
    return item, container, _cron_job(kube, name, job_name, deadline, log)


def _single_config_key_projection(
    workload: dict[str, Any],
    container: dict[str, Any],
    configmap: str,
    key: str,
) -> None:
    mount = find_mount(container, "/srv")
    require(mount is not None, "input mount /srv is absent")
    volume = find_volume(pod_spec(workload), mount.get("name", ""))
    require(volume is not None, "input volume is absent")
    if "configMap" in volume:
        source = volume["configMap"]
    else:
        sources = volume.get("projected", {}).get("sources", [])
        config_sources = [item["configMap"] for item in sources if set(item) == {"configMap"}]
        require(len(sources) == len(config_sources) == 1, "input projection contains another source")
        source = config_sources[0]
    require(source.get("name") == configmap, "input ConfigMap differs")
    require(source.get("items") == [{"key": key, "path": "index.html"}], f"input must project only {key}")


def _selector_matches(labels: dict[str, str], selector: dict[str, Any]) -> bool:
    if any(labels.get(key) != value for key, value in selector.get("matchLabels", {}).items()):
        return False
    for expression in selector.get("matchExpressions", []):
        key, operator, values = expression.get("key"), expression.get("operator"), set(expression.get("values", []))
        present = key in labels
        if operator == "In" and (not present or labels[key] not in values): return False
        if operator == "NotIn" and present and labels[key] in values: return False
        if operator == "Exists" and not present: return False
        if operator == "DoesNotExist" and present: return False
    return True


def _restricted_egress(
    kube: Kubectl,
    selected_labels: dict[str, str],
    allowed_dependencies: list[tuple[dict[str, str], set[tuple[str, int | str]]]],
    *,
    dns: bool = True,
) -> str:
    """Check the additive egress union while accepting arbitrary policy splitting.

    This is intentionally conservative: it rejects widened non-DNS paths, but does not
    prescribe policy names, exact selector syntax, or a particular DNS peer encoding.
    """

    policies = []
    for item in kube.list("networkpolicies"):
        spec = item.get("spec", {})
        if _selector_matches(selected_labels, spec.get("podSelector", {})) and (
            "Egress" in spec.get("policyTypes", []) or "egress" in spec
        ):
            policies.append(item)
    require(policies, f"no egress isolation selects {selected_labels}")
    saw_dependency = [False] * len(allowed_dependencies)
    dns_protocols: set[str] = set()
    for policy in policies:
        for rule in policy.get("spec", {}).get("egress", []):
            raw_ports = rule.get("ports", [])
            require(raw_ports, "an unbounded egress rule is present")
            ports = {(entry.get("protocol", "TCP"), entry.get("port")) for entry in raw_ports}
            if ports and all(str(port) == "53" and protocol in {"UDP", "TCP"} for protocol, port in ports):
                require(dns, "DNS egress is not declared for this workload")
                dns_protocols.update(protocol for protocol, _ in ports)
                continue
            peers = rule.get("to", [])
            require(peers, "non-DNS egress has an unrestricted destination")
            matched = False
            for index, (labels, allowed_ports) in enumerate(allowed_dependencies):
                if ports.issubset(allowed_ports) and all(
                    set(peer).issubset({"podSelector", "namespaceSelector"})
                    and _selector_matches(labels, peer.get("podSelector", {}))
                    and bool(peer.get("podSelector"))
                    for peer in peers
                ):
                    saw_dependency[index] = True
                    matched = True
                    break
            require(matched, f"egress rule widens the public dependency/port contract: {ports}")
    require(all(saw_dependency), "a required dependency egress path is absent")
    if dns:
        require({"UDP", "TCP"}.issubset(dns_protocols), "DNS egress must allow both UDP and TCP 53")
    return "egress is isolated to the declared dependency paths and DNS"


def _hard6_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def origin() -> str:
        dep = _workload_image_and_count(kube, "deployment", "release-origin", "app=release-origin", 2)
        require(pod_labels(dep).get("app") == "release-origin", "origin labels differ")
        _token_policy(dep, False); container = _container(dep, "origin")
        _mount_source(dep, container, "/source", "configMap", "mirror-source")
        _port(container, "origin-http", 8080)
        require(container.get("startupProbe", {}).get("httpGet", {}).get("path") == "/", "startup probe differs")
        require(container.get("startupProbe", {}).get("periodSeconds", 10) == 5, "startup probe period differs")
        # The public task fixes the path and cadence but intentionally does not
        # prescribe a readiness failure threshold.
        _probe(container, "readinessProbe", "/", "origin-http", 5, None)
        return "origin wiring and two Ready replicas match"

    def mirror() -> str:
        dep = _workload_image_and_count(kube, "deployment", "release-mirror", "app=release-mirror", 2)
        _token_policy(dep, False); init = _sole_container(dep, init=True)
        require("release-origin-svc" in command_text(init), "init does not gate on origin Service")
        _bounded_attempts(command_text(init), 45)
        main = _container(dep, "mirror"); _secret_env(main, "TOKEN", "mirror-key", "TOKEN")
        _emptydir_mount(dep, main, "/www"); _port(main, "mirror-http", 8081)
        _probe(main, "readinessProbe", "/", "mirror-http", 5, 2)
        require("TOKEN" in command_text(main), "mirror command does not use the projected token")
        return "mirror has a real source/token gate and two Ready replicas"

    def smoke() -> str:
        workload, container, result = _one_job(kube, "release-smoke", 150, "release-smoke-ok\n")
        require(pod_labels(workload).get("access") == "release", "smoke access label differs")
        text = command_text(container)
        require("release-mirror-svc" in text and "mirror:release=v1" in text, "smoke does not check the exact mirror response")
        _bounded_attempts(text, 60)
        return result

    def live_update() -> str:
        return _update_configmap(
            kube, "mirror-source", "index.html", "release=v2\n", "release=v1\n",
            lambda value: _all_pod_bodies(kube, "app=release-origin", "origin", 8080, value, 2)
            and _all_pod_bodies(kube, "app=release-mirror", "mirror", 8081, f"mirror:{value}", 2),
            150, [("deployment", "release-origin"), ("deployment", "release-mirror")],
            ["app=release-origin", "app=release-mirror"],
        )

    def source_outage() -> str:
        result = _service_outage(kube, "release-origin-svc", lambda: _ready_count(kube,"app=release-mirror") == 0 and _endpoint_count(kube, "release-mirror-svc") == 0, lambda: _endpoint_count(kube, "release-origin-svc") == 2 and _endpoint_count(kube, "release-mirror-svc") == 2, 90, 180)
        _body_probe(kube, "aipc-eval-mirror-restored", "http://release-mirror-svc", "mirror:release=v1\n", labels={"access":"release"})
        return result + "; exact mirror response recovered"

    return [
        ("R01", "namespace and scope", lambda: _scope(kube, "hard-mirror-ns", candidate_path)),
        ("R02", "source content", lambda: _exact_configmap(kube, "mirror-source", {"index.html": "release=v1\n"})),
        ("R03", "immutable token", lambda: _secret(kube, "mirror-key", "TOKEN", "mirror-ok")),
        ("R04", "origin deployment", origin),
        ("R05", "origin service", lambda: (_service(kube, "release-origin-svc", {"app": "release-origin"}, 8080, "origin-http") and _body_probe(kube, "aipc-eval-origin", "http://release-origin-svc:8080", "release=v1\n", labels={"role": "mirror"}))),
        ("R06", "mirror deployment", mirror),
        ("R07", "mirror service", lambda: (_service(kube, "release-mirror-svc", {"app": "release-mirror"}, 80, "mirror-http") and _body_probe(kube, "aipc-eval-mirror", "http://release-mirror-svc", "mirror:release=v1\n", labels={"access": "release"}))),
        ("R08", "pod hardening", lambda: _harden_with_writes(kube, [("deployment", "release-origin", set()), ("deployment", "release-mirror", {"/www"})])),
        ("R09", "network isolation", lambda: (_live_ingress_matrix(kube, [("aipc-eval-origin-ok", "http://release-origin-svc:8080", {"role": "mirror"}), ("aipc-eval-origin-access", "http://release-origin-svc:8080", {"access":"release"}), ("aipc-eval-mirror-ok", "http://release-mirror-svc", {"access": "release"})], [("aipc-eval-origin-deny", "http://release-origin-svc:8080"), ("aipc-eval-mirror-deny", "http://release-mirror-svc")]) and _restricted_egress(kube,{"app":"release-origin"},[],dns=False) and _restricted_egress(kube,{"app":"release-mirror"},[({"app":"release-origin"},{("TCP",8080),("TCP","origin-http")})]))),
        ("R10", "availability budgets", lambda: _pdb_set(kube, [("release-origin-pdb", "release-origin", 1), ("release-mirror-pdb", "release-mirror", 1)])),
        ("R11", "bounded smoke Job", smoke),
        ("R12", "resource governance", lambda: _limits(kube, "mirror-defaults", "mirror-quota", {"pods":"12","services":"4","count/jobs.batch":"3","requests.cpu":"2","requests.memory":"1Gi","limits.cpu":"4","limits.memory":"2Gi"})),
        ("R13", "live source propagation", live_update),
        ("R14", "source Service outage", source_outage),
    ]


def _hard7_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def prefix() -> str:
        _exact_configmap(kube, "shard-prefix", {"PREFIX": "shard"})
        return _update_configmap(
            kube, "shard-prefix", "PREFIX", "shard2", "shard",
            lambda value: all(
                _local_body(kube, f"shard-{ordinal}", _sole_container(kube.get("statefulset", "shard"))["name"], 8080, f"{value}-shard-{ordinal}-signed-g1\n")
                for ordinal in range(2)
            ) and _all_pod_bodies(kube, "app=shard-index", _sole_container(kube.get("deployment", "shard-index"))["name"], 8081, f"{value}-ready=2\n", 2),
            180, [("statefulset", "shard"), ("deployment", "shard-index")], ["app=shard", "app=shard-index"],
        )

    def stateful() -> str:
        sts = _workload_image_and_count(kube, "statefulset", "shard", "app=shard", 2)
        _stateful_controls(sts, "shard-headless", 2)
        require(pod_labels(sts).get("component") == "member", "member label differs")
        _token_policy(sts, False); init = _container(sts, "initialize", init=True)
        require("/data/generation" in command_text(init), "initializer does not preserve generation")
        require(find_mount(init, "/data") is not None and find_mount(init, "/data").get("name") == "data", "initializer claim mount differs")
        main = _sole_container(sts); _secret_input(sts, main, "SIGN", "shard-signing", "SIGN")
        _mount_source(sts, main, "/config", "configMap", "shard-prefix", read_only=True)
        require(find_mount(main, "/data") is not None and find_mount(main, "/data").get("name") == "data", "member claim mount differs")
        _port(main, "shard-http", 8080); _probe(main, "readinessProbe", "/", "shard-http", 5, 3)
        return "StatefulSet identity, retained claims, initialization and readiness match"

    def index() -> str:
        dep = _workload_image_and_count(kube, "deployment", "shard-index", "app=shard-index", 2)
        _deployment_controls(dep, replicas=2, min_ready=5, deadline=210); _token_policy(dep, False)
        main = _sole_container(dep); _emptydir_mount(dep, main, "/www")
        _mount_source(dep, main, "/config", "configMap", "shard-prefix", read_only=True)
        require("shard-0.shard-headless" in command_text(main) and "shard-1.shard-headless" in command_text(main), "index does not require both stable members")
        _port(main, "index-http", 8081); _probe(main, "readinessProbe", "/", "index-http", 5, 2)
        return "index checks both stable members and has two Ready replicas"

    def cron() -> str:
        item = kube.get("cronjob", "shard-audit"); spec = item.get("spec", {})
        require(spec.get("suspend") is True and spec.get("schedule") == "17 * * * *", "audit schedule/suspension differs")
        require(spec.get("concurrencyPolicy") == "Forbid" and spec.get("successfulJobsHistoryLimit") == 1 and spec.get("failedJobsHistoryLimit") == 1, "audit lifecycle differs")
        pod = _cron_spec(item); require(pod.get("restartPolicy") == "Never", "audit restartPolicy differs")
        require(item["spec"]["jobTemplate"]["spec"].get("activeDeadlineSeconds") == 210, "audit deadline differs")
        container = _one_cron_container(item)
        require(container.get("image") == "busybox:1.36.1", "audit image differs")
        text = command_text(container)
        require("shard-index-svc" in text and "shard-ready=2" in text, "audit does not check the exact index body")
        _bounded_attempts(text, 90)
        return _cron_job(kube, "shard-audit", "aipc-eval-shard-audit", 210, "shard-audit-ok\n")

    def members() -> str:
        _service(kube, "shard-svc", {"app":"shard"}, 80, "shard-http")
        require(_endpoint_count(kube, "shard-svc") == 2, "member endpoint count differs")
        container = _sole_container(kube.get("statefulset", "shard"))["name"]
        for ordinal in range(2):
            require(_local_body(kube, f"shard-{ordinal}", container, 8080, f"shard-shard-{ordinal}-signed-g1\n"), f"shard-{ordinal} body differs")
        return "both stable members have exact bodies"

    def replacement() -> str:
        pod = kube.get("pod", "shard-1"); old = pod["metadata"]["uid"]
        container = _sole_container(kube.get("statefulset", "shard"))["name"]
        kube.exec("shard-1", "printf 'retained\\n' > /data/oracle-marker", container)
        try:
            kube.run(["delete", "pod", "shard-1", "-n", kube.namespace, "--wait=true"])
            kube.wait_until(lambda: kube.get("pod", "shard-1")["metadata"]["uid"] != old and _ready(kube.get("pod", "shard-1")), "different-UID shard-1", 150, 2)
            kube.exec("shard-1", "test \"$(cat /data/oracle-marker)\" = retained && test \"$(cat /data/generation)\" = g1", container)
            kube.wait_until(lambda: _endpoint_count(kube, "shard-svc") == 2 and _endpoint_count(kube, "shard-index-svc") == 2, "member and index endpoint recovery", 210, 2)
            _body_probe(kube, "aipc-eval-shard-index-restored", "http://shard-index-svc", "shard-ready=2\n", labels={"access":"shard"})
        finally:
            kube.run(["exec", "-n", kube.namespace, "shard-1", "-c", container, "--", "rm", "-f", "/data/oracle-marker"], check=False)
        return "shard-1 retained marker/generation and both Services recovered"

    return [
        ("R01", "namespace and scope", lambda: _scope(kube, "hard-shard-ns", candidate_path)),
        ("R02", "prefix and live propagation", prefix),
        ("R03", "immutable signing Secret", lambda: _secret(kube, "shard-signing", "SIGN", "signed")),
        ("R04", "headless Service", lambda: (_service(kube, "shard-headless", {"app":"shard"}, 8080, "shard-http", headless=True) and "headless Service is exact")),
        ("R05", "retained StatefulSet", stateful),
        ("R06", "member Service and ordinals", members),
        ("R07", "index Deployment", index),
        ("R08", "index Service", lambda: (_service(kube, "shard-index-svc", {"app":"shard-index"}, 80, "index-http") and _body_probe(kube, "aipc-eval-shard-index", "http://shard-index-svc", "shard-ready=2\n", labels={"access":"shard"}))),
        ("R09", "pod hardening", lambda: _harden_with_writes(kube, [("statefulset","shard",{"/data"}),("deployment","shard-index",{"/www"}),("cronjob","shard-audit",set())])),
        ("R10", "network isolation", lambda: (_live_ingress_matrix(kube, [("aipc-eval-shard-member-index","http://shard-svc",{"component":"index"}),("aipc-eval-shard-member-audit","http://shard-svc",{"role":"shard-audit"}),("aipc-eval-shard-member-ok","http://shard-svc",{"access":"shard"}),("aipc-eval-shard-index-audit","http://shard-index-svc",{"role":"shard-audit"}),("aipc-eval-shard-index-ok","http://shard-index-svc",{"access":"shard"})], [("aipc-eval-shard-member-deny","http://shard-svc"),("aipc-eval-shard-index-deny","http://shard-index-svc")]) and _restricted_egress(kube,{"app":"shard","component":"member"},[],dns=False) and _restricted_egress(kube,{"app":"shard-index","component":"index"},[({"app":"shard","component":"member"},{("TCP",8080),("TCP","shard-http")})]))),
        ("R11", "availability budgets", lambda: _pdb_set(kube, [("shard-pdb","shard",1),("shard-index-pdb","shard-index",1)])),
        ("R12", "exact quota", lambda: _quota_only(kube, "shard-quota", {"pods":"12","persistentvolumeclaims":"3","requests.storage":"256Mi","services":"4","count/jobs.batch":"3","count/cronjobs.batch":"2","requests.cpu":"2","requests.memory":"1Gi","limits.cpu":"4","limits.memory":"2Gi"})),
        ("R13", "executable audit template", cron),
        ("R14", "retained member replacement", replacement),
    ]


def _hard8_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def account() -> str:
        item = kube.get("serviceaccount", "catalog-reader")
        require(item.get("automountServiceAccountToken") is True, "catalog-reader automount must be true")
        return "catalog-reader is explicitly token-enabled"

    def role() -> str:
        _role(kube, "catalog-page-reader", {("", "configmaps", "catalog-page", "get")})
        return "Role grants only get on catalog-page"

    def binding() -> str:
        _binding(kube, "catalog-page-reader-binding", "catalog-page-reader", "catalog-reader")
        return "RoleBinding has the exact sole subject and roleRef"

    def deployment() -> str:
        dep = _workload_image_and_count(kube, "deployment", "authorized-catalog", "app=authorized-catalog", 2)
        _deployment_controls(dep, replicas=2, min_ready=5, deadline=180)
        require(pod_spec(dep).get("serviceAccountName") == "catalog-reader", "ServiceAccount differs")
        _token_policy(dep, True); main = _container(dep, "catalog")
        _secret_env(main, "TOKEN", "catalog-token", "TOKEN")
        _mount_source(dep, main, "/srv", "configMap", "catalog-page", read_only=True)
        _emptydir_mount(dep, main, "/work"); _port(main, "catalog-http", 8080)
        probe = main.get("readinessProbe", {}).get("exec", {}).get("command", [])
        probe_text = " ".join(probe)
        require(probe and "/work/authorized" in probe_text and ("127.0.0.1" in probe_text or "localhost" in probe_text) and "catalog=v1" in probe_text, "readiness does not require authorization and exact local content")
        text = command_text(main)
        require(re.search(r"https://[^ ]+/api/v1/namespaces/[^ ]+/configmaps/catalog-page", text) is not None and "authorization" in text.lower() and "TOKEN" in text, "narrow authenticated API check is absent")
        _bounded_fetch(text, 5)
        return "catalog has a live API authorization gate and two Ready replicas"

    def live_update() -> str:
        return _update_configmap(
            kube, "catalog-page", "index.html", "catalog=v2\n", "catalog=v1\n",
            lambda value: _all_pod_bodies(kube, "app=authorized-catalog", "catalog", 8080, value, 2),
            150, [("deployment", "authorized-catalog")], ["app=authorized-catalog"],
        )

    def auth_outage() -> str:
        saved = _snapshot(kube, "rolebinding", "catalog-page-reader-binding")
        try:
            kube.run(["delete", "rolebinding", "catalog-page-reader-binding", "-n", kube.namespace, "--wait=true"])
            kube.wait_until(lambda: _ready_count(kube,"app=authorized-catalog") == 0 and _endpoint_count(kube, "authorized-catalog-svc") == 0, "catalog authorization readiness loss", 90, 2)
        finally:
            try:
                kube.run(["apply", "-f", "-"], input_text=json.dumps(saved))
            except KubectlError as exc:
                raise EvaluationInfrastructureError("could not restore catalog RoleBinding") from exc
            kube.wait_until(lambda: _endpoint_count(kube, "authorized-catalog-svc") == 2, "catalog authorization recovery", 180, 2)
        _body_probe(kube, "aipc-eval-catalog-auth-restored", "http://authorized-catalog-svc", "catalog=v1\n", labels={"access":"catalog"})
        return "authorization loss removed readiness and exact restoration recovered it"

    def audit() -> str:
        workload, container, result = _one_job(kube, "catalog-audit", 150, "catalog-audit-ok\n")
        require(pod_spec(workload).get("serviceAccountName") == "catalog-reader", "audit ServiceAccount differs")
        require(pod_labels(workload).get("access") == "catalog", "audit access label differs")
        text = command_text(container)
        require(re.search(r"https://[^ ]+/api/v1/namespaces/[^ ]+/configmaps/catalog-page", text) is not None and "authorization" in text.lower() and "authorized-catalog-svc" in text and "catalog=v1" in text, "audit omits the narrow authenticated or Service check")
        _bounded_fetch(text, 5); _bounded_attempts(text, 60)
        return result

    return [
        ("R01", "namespace and scope", lambda: _scope(kube, "hard-catalog-ns", candidate_path)),
        ("R02", "catalog page", lambda: _exact_configmap(kube, "catalog-page", {"index.html":"catalog=v1\n"})),
        ("R03", "catalog token", lambda: _secret(kube, "catalog-token", "TOKEN", "catalog-key")),
        ("R04", "reader identity", account),
        ("R05", "least-privilege Role", role),
        ("R06", "exact RoleBinding", binding),
        ("R07", "authorization-gated Deployment", deployment),
        ("R08", "catalog Service", lambda: (_service(kube, "authorized-catalog-svc", {"app":"authorized-catalog"}, 80, "catalog-http") and _body_probe(kube, "aipc-eval-catalog", "http://authorized-catalog-svc", "catalog=v1\n", labels={"access":"catalog"}))),
        ("R09", "authenticated audit Job", audit),
        ("R10", "pod hardening", lambda: _harden_with_writes(kube, [("deployment","authorized-catalog",{"/work"}),("job","catalog-audit",set())])),
        ("R11", "network isolation", lambda: _live_ingress_matrix(kube, [("aipc-eval-catalog-ok","http://authorized-catalog-svc",{"access":"catalog"})], [("aipc-eval-catalog-deny","http://authorized-catalog-svc")])),
        ("R12", "availability budget", lambda: (_pdb(kube, "catalog-pdb", {"app":"authorized-catalog"}, 1) or "catalog PDB is exact")),
        ("R13", "live catalog projection", live_update),
        ("R14", "authorization loss and recovery", auth_outage),
    ]


def _hard9_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def seed() -> str:
        workload, container, result = _one_job(kube, "artifact-seed", 90, "artifact-seed-ok\n")
        _mount_source(workload, container, "/data", "persistentVolumeClaim", "artifact-data")
        _mount_source(workload, container, "/config", "configMap", "artifact-settings", read_only=True)
        _secret_input(workload, container, "SIGN", "artifact-signing", "SIGN")
        text = command_text(container)
        require(all(value in text for value in ("/data/artifact", "/config/NAME", "/config/VERSION")), "seed does not compose the declared inputs")
        require("mv" in text and ("if [ ! -e" in text or "if [ ! -f" in text or "test -e" in text or "test -f" in text), "seed is not visibly atomic/idempotent")
        return result

    def server() -> str:
        dep = _workload_image_and_count(kube, "deployment", "artifact-server", "app=artifact-server", 1)
        _recreate(dep); _token_policy(dep, False)
        container = _sole_container(dep)
        _mount_source(dep, container, "/srv", "persistentVolumeClaim", "artifact-data", read_only=False)
        _port(container, "artifact-http", 8080)
        probe = container.get("readinessProbe", {})
        require(probe.get("periodSeconds", 10) == 5 and probe.get("failureThreshold", 3) == 2, "server readiness timing differs")
        require("/srv/artifact" in " ".join(probe.get("exec", {}).get("command", [])), "readiness is not tied to the durable artifact")
        return "artifact-server uses the retained claim and has one Ready replica"

    def verify() -> str:
        workload, container, result = _one_job(kube, "artifact-verify", 120, "artifact-verify-ok\n")
        require(pod_labels(workload).get("access") == "artifact", "verify access label differs")
        text = command_text(container)
        expected = "daily-v1-sealed\n"
        encodings = {
            expected.rstrip("\n"),
            base64.b64encode(expected.encode()).decode(),
            expected.encode().hex(),
        }
        require(
            "artifact-svc" in text
            and "/artifact" in text
            and any(value in text for value in encodings),
            "verify does not check the exact Service artifact",
        )
        _bounded_attempts(text, 45)
        return result

    def hardening() -> str:
        _harden_with_writes(kube, [("job", "artifact-seed", {"/data"}), ("deployment", "artifact-server", {"/srv"}), ("job", "artifact-verify", set())])
        seed_job = _as_job_workload(kube.get("job", "artifact-seed")); server_dep = kube.get("deployment", "artifact-server")
        require(find_mount(_sole_container(seed_job), "/data").get("readOnly", False) is False, "seed data claim must be writable")
        require(find_mount(_sole_container(server_dep), "/srv").get("readOnly", False) is False, "server claim must be writable for evaluator marker")
        return "all containers are hardened and only declared claim paths are writable"

    def replacement() -> str:
        pods = [pod for pod in kube.list("pods", "app=artifact-server") if _ready(pod)]
        require(len(pods) == 1, "artifact replacement requires one Ready server")
        pod = pods[0]; name = pod["metadata"]["name"]; old_uid = pod["metadata"]["uid"]
        container = _sole_container(kube.get("deployment", "artifact-server"))["name"]
        kube.exec(name, "printf 'survivor\\n' > /srv/survivor", container)
        try:
            kube.run(["delete", "pod", name, "-n", kube.namespace, "--wait=true"])
            kube.wait_until(lambda: any(_ready(item) and item["metadata"]["uid"] != old_uid for item in kube.list("pods", "app=artifact-server")), "different-UID artifact server", 150, 2)
            new_pod = next(item for item in kube.list("pods", "app=artifact-server") if _ready(item) and item["metadata"]["uid"] != old_uid)
            kube.exec(new_pod["metadata"]["name"], "test \"$(cat /srv/artifact)\" = daily-v1-sealed && test \"$(cat /srv/survivor)\" = survivor", container)
            kube.wait_until(lambda: _endpoint_count(kube, "artifact-svc") == 1, "artifact endpoint recovery", 180, 2)
            _body_probe(kube, "aipc-eval-artifact-recovery", "http://artifact-svc/artifact", "daily-v1-sealed\n", labels={"access":"artifact"})
        finally:
            for item in kube.list("pods", "app=artifact-server"):
                kube.run(["exec", "-n", kube.namespace, item["metadata"]["name"], "-c", container, "--", "rm", "-f", "/srv/survivor"], check=False)
        return "artifact and evaluator marker survived a different-UID replacement"

    return [
        ("R01", "namespace and scope", lambda: _scope(kube, "hard-artifact-ns", candidate_path)),
        ("R02", "immutable settings", lambda: _immutable_configmap(kube, "artifact-settings", {"NAME":"daily", "VERSION":"v1"})),
        ("R03", "immutable signing Secret", lambda: _secret(kube, "artifact-signing", "SIGN", "sealed")),
        ("R04", "durable claim", lambda: _claim(kube, "artifact-data", "96Mi")),
        ("R05", "idempotent seed Job", seed),
        ("R06", "artifact server", server),
        ("R07", "artifact Service", lambda: (_service(kube, "artifact-svc", {"app":"artifact-server"}, 80, "artifact-http") and _body_probe(kube, "aipc-eval-artifact", "http://artifact-svc/artifact", "daily-v1-sealed\n", labels={"access":"artifact"}))),
        ("R08", "bounded verifier Job", verify),
        ("R09", "pod hardening", hardening),
        ("R10", "network isolation", lambda: _live_ingress_matrix(kube, [("aipc-eval-artifact-ok", "http://artifact-svc/artifact", {"access":"artifact"})], [("aipc-eval-artifact-deny", "http://artifact-svc/artifact")])),
        ("R11", "availability budget", lambda: (_pdb(kube, "artifact-pdb", {"app":"artifact-server"}, 1) or "artifact PDB is exact")),
        ("R12", "resource governance", lambda: _limits(kube, "artifact-defaults", "artifact-quota", {"pods":"8","persistentvolumeclaims":"2","requests.storage":"256Mi","services":"2","count/jobs.batch":"4","requests.cpu":"1","requests.memory":"512Mi","limits.cpu":"2","limits.memory":"1Gi"})),
        ("R13", "durable replacement", replacement),
    ]


def _hard10_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def agent() -> str:
        ds = _workload_image_and_count(kube, "daemonset", "node-status", "app=node-status", 1)
        _labels_include(ds, {"app":"node-status", "component":"agent"})
        spec = pod_spec(ds); require(not spec.get("nodeSelector"), "agent must not constrain nodes")
        require(not spec.get("tolerations"), "agent template has a non-default toleration")
        _token_policy(ds, False); container = _sole_container(ds)
        _mount_source(ds, container, "/srv", "configMap", "node-page", read_only=True)
        _port(container, "agent-http", 8090); _probe(container, "readinessProbe", "/", "agent-http", 5, None)
        return "one unconstrained node agent is Ready with exact wiring"

    def agent_replacement() -> str:
        pods = [pod for pod in kube.list("pods", "app=node-status") if _ready(pod)]
        require(len(pods) == 1, "agent replacement requires exactly one Ready pod")
        old = pods[0]["metadata"]["uid"]
        kube.run(["delete", "pod", pods[0]["metadata"]["name"], "-n", kube.namespace, "--wait=true"])
        kube.wait_until(lambda: any(_ready(p) and p["metadata"]["uid"] != old for p in kube.list("pods", "app=node-status")), "different-UID agent", 120, 2)
        _body_probe(kube, "aipc-eval-agent-replacement", "http://node-status-svc:8090", "node=v1\n", labels={"component":"registry"})
        return "DaemonSet restored a different-UID Ready agent and exact body"

    def account() -> str:
        item = kube.get("serviceaccount", "registry-reader")
        require(item.get("automountServiceAccountToken") is True, "registry-reader token automount must be true")
        return "registry-reader is explicitly token-enabled"

    def role() -> str:
        return _exact_role_access(kube, "node-status-reader", {("apps","daemonsets","node-status","get"),("apps","daemonsets/status","node-status","get")})

    def registry() -> str:
        dep = _workload_image_and_count(kube, "deployment", "node-registry", "app=node-registry", 2)
        _labels_include(dep, {"app":"node-registry", "component":"registry"})
        _deployment_controls(dep, replicas=2, min_ready=5, deadline=180)
        require(pod_spec(dep).get("serviceAccountName") == "registry-reader", "registry ServiceAccount differs")
        container = _sole_container(dep); _secret_input(dep, container, "TOKEN", "registry-key", "TOKEN")
        _emptydir_mount(dep, container, "/www"); _port(container, "registry-http", 8080)
        _probe(container, "readinessProbe", "/", "registry-http", 5, 2)
        text = command_text(container); _bounded_fetch(text, 5)
        require(re.search(r"https://[^ ]+/apis/apps/v1/namespaces/[^ ]+/daemonsets/node-status(?:/status)?", text) is not None and "authorization" in text.lower() and "TOKEN" in text and "node-status-svc" in text and "/www/index.html" in text and "rm" in text, "registry does not enforce both narrow live gates")
        return "two Ready registries implement the bounded API and Service gates"

    def binding_outage() -> str:
        saved = _snapshot(kube, "rolebinding", "node-status-reader-binding")
        try:
            kube.run(["delete", "rolebinding", "node-status-reader-binding", "-n", kube.namespace, "--wait=true"])
            kube.wait_until(lambda: _ready_count(kube,"app=node-registry") == 0 and _endpoint_count(kube, "node-registry-svc") == 0, "registry authorization readiness loss", 90, 2)
        finally:
            try: kube.run(["apply", "-f", "-"], input_text=json.dumps(saved))
            except KubectlError as exc: raise EvaluationInfrastructureError("could not restore registry RoleBinding") from exc
            kube.wait_until(lambda: _endpoint_count(kube, "node-status-svc") == 1 and _endpoint_count(kube, "node-registry-svc") == 2, "registry authorization recovery", 180, 2)
        _body_probe(kube, "aipc-eval-registry-restored", "http://node-registry-svc", "registry:node=v1\n", labels={"access":"registry"})
        return "RoleBinding loss removed readiness and exact restoration recovered it"

    def live_update() -> str:
        return _update_configmap(kube, "node-page", "index.html", "node=v2\n", "node=v1\n", lambda v: _all_pod_bodies(kube,"app=node-status",_sole_container(kube.get("daemonset","node-status"))["name"],8090,v,1) and _all_pod_bodies(kube,"app=node-registry",_sole_container(kube.get("deployment","node-registry"))["name"],8080,f"registry:{v}",2), 180, [("daemonset","node-status"),("deployment","node-registry")], ["app=node-status","app=node-registry"])

    return [
        ("R01", "namespace and scope", lambda: _scope(kube, "hard-registry-ns", candidate_path)),
        ("R02", "mutable node page", lambda: _exact_configmap(kube, "node-page", {"index.html":"node=v1\n"})),
        ("R03", "immutable registry key", lambda: _secret(kube, "registry-key", "TOKEN", "registry-ok")),
        ("R04", "node agent and replacement", lambda: (agent() and agent_replacement())),
        ("R05", "agent Service", lambda: (_service(kube, "node-status-svc", {"app":"node-status"}, 8090, "agent-http") and _body_probe(kube, "aipc-eval-agent", "http://node-status-svc:8090", "node=v1\n", labels={"component":"registry"}))),
        ("R06", "registry identity", account),
        ("R07", "least-privilege Role", role),
        ("R08", "RoleBinding authorization transition", lambda: (_binding(kube, "node-status-reader-binding", "node-status-reader", "registry-reader") or binding_outage())),
        ("R09", "dual-gated registry", registry),
        ("R10", "registry Service", lambda: (_service(kube, "node-registry-svc", {"app":"node-registry"}, 80, "registry-http") and _body_probe(kube, "aipc-eval-registry", "http://node-registry-svc", "registry:node=v1\n", labels={"access":"registry"}))),
        ("R11", "pod hardening", lambda: _harden_with_writes(kube, [("daemonset","node-status",set()),("deployment","node-registry",{"/www"})])),
        ("R12", "network isolation", lambda: _live_ingress_matrix(kube, [("aipc-eval-agent-registry","http://node-status-svc:8090",{"component":"registry"}),("aipc-eval-agent-access","http://node-status-svc:8090",{"access":"registry"}),("aipc-eval-registry-access","http://node-registry-svc",{"access":"registry"})], [("aipc-eval-agent-deny","http://node-status-svc:8090"),("aipc-eval-registry-deny","http://node-registry-svc")])),
        ("R13", "availability budget", lambda: (_pdb(kube, "node-registry-pdb", {"app":"node-registry"}, 1) or "registry PDB is exact")),
        ("R14", "live page propagation", live_update),
    ]


def _hard11_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def deployment() -> str:
        dep = _workload_image_and_count(kube, "deployment", "secure-bulletin", "app=secure-bulletin", 2)
        _deployment_controls(dep, replicas=2, min_ready=5, deadline=150); _token_policy(dep, False)
        container = _sole_container(dep)
        _mount_source(dep, container, "/page", "configMap", "bulletin-page", read_only=True)
        _mount_source(dep, container, "/code", "secret", "bulletin-code", read_only=True)
        _emptydir_mount(dep, container, "/www"); _port(container, "bulletin-http", 8080)
        _probe(container, "readinessProbe", "/", "bulletin-http", 5, 2)
        text = command_text(container)
        require("/page" in text and "/code" in text and "/www/index.html" in text and "sleep 2" in text and "rm" in text, "bulletin does not implement the projected-input gate")
        return "two Ready bulletin pods combine both projected inputs"

    def cron() -> str:
        item, container, result = _cron_contract(kube, "bulletin-check", "*/7 * * * *", 150, "aipc-eval-bulletin-check", "bulletin-check-ok\n")
        require(pod_labels({"spec":{"template":item["spec"]["jobTemplate"]["spec"]["template"]}}).get("role") == "bulletin-check", "checker role label differs")
        require(container.get("image") == "busybox:1.36.1", "checker image differs")
        text = command_text(container); require("secure-bulletin-svc" in text and "bulletin=v1;code=blue" in text, "checker does not require the exact body")
        _bounded_attempts(text, 60)
        return result

    def secret_update() -> str:
        return _secret_projection(kube, "bulletin-code", "code", "green", "blue", lambda v: _all_pod_bodies(kube,"app=secure-bulletin",_sole_container(kube.get("deployment","secure-bulletin"))["name"],8080,f"bulletin=v1;code={v}\n",2), [("deployment","secure-bulletin")], ["app=secure-bulletin"], 150)

    def lifecycle() -> str:
        dep = kube.get("deployment", "secure-bulletin"); spec = dep.get("spec", {})
        require(spec.get("revisionHistoryLimit") == 2, "revisionHistoryLimit differs")
        require(pod_spec(dep).get("terminationGracePeriodSeconds") == 10, "termination grace differs")
        container = _sole_container(dep)
        require(container.get("lifecycle", {}).get("preStop", {}).get("exec", {}).get("command") == ["sh","-c","rm -f /www/index.html; sleep 3"], "preStop command differs")
        return "revision retention, grace period and readiness-dropping preStop are exact"

    return [
        ("R01", "namespace and scope", lambda: _scope(kube, "hard-bulletin-ns", candidate_path)),
        ("R02", "immutable bulletin page", lambda: _immutable_configmap(kube, "bulletin-page", {"index.html":"bulletin=v1\n"})),
        ("R03", "mutable bulletin code", lambda: _mutable_secret(kube, "bulletin-code", "code", "blue")),
        ("R04", "projected bulletin Deployment", deployment),
        ("R05", "bulletin Service", lambda: (_service(kube, "secure-bulletin-svc", {"app":"secure-bulletin"}, 80, "bulletin-http") and _body_probe(kube, "aipc-eval-bulletin", "http://secure-bulletin-svc", "bulletin=v1;code=blue\n", labels={"access":"bulletin"}))),
        ("R06", "pod hardening", lambda: _harden_with_writes(kube, [("deployment","secure-bulletin",{"/www"}),("cronjob","bulletin-check",set())])),
        ("R07", "network isolation", lambda: (_live_ingress_matrix(kube, [("aipc-eval-bulletin-access","http://secure-bulletin-svc",{"access":"bulletin"}),("aipc-eval-bulletin-checker","http://secure-bulletin-svc",{"role":"bulletin-check"})], [("aipc-eval-bulletin-deny","http://secure-bulletin-svc")]) and _restricted_egress(kube, {"app":"secure-bulletin"}, [], dns=True))),
        ("R08", "availability budget", lambda: (_pdb(kube, "bulletin-pdb", {"app":"secure-bulletin"}, 1) or "bulletin PDB is exact")),
        ("R09", "resource governance", lambda: _limits(kube, "bulletin-defaults", "bulletin-quota", {"pods":"10","services":"3","secrets":"3","count/jobs.batch":"3","count/cronjobs.batch":"2","requests.cpu":"2","requests.memory":"1Gi","limits.cpu":"4","limits.memory":"2Gi"})),
        ("R10", "executable checker template", cron),
        ("R11", "live Secret projection", secret_update),
        ("R12", "replacement continuity", lambda: _replacement_continuity(kube, "app=secure-bulletin", "secure-bulletin-svc", "http://secure-bulletin-svc", "bulletin=v1;code=blue\n", {"access":"bulletin"}, 90)),
        ("R13", "termination lifecycle", lifecycle),
    ]


def _hard12_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def stateful() -> str:
        sts = _workload_image_and_count(kube, "statefulset", "journal", "app=journal", 3)
        spec = sts.get("spec", {})
        require(spec.get("serviceName") == "journal-headless" and spec.get("podManagementPolicy") == "Parallel", "journal identity/management differs")
        require(spec.get("persistentVolumeClaimRetentionPolicy") == {"whenDeleted":"Retain","whenScaled":"Retain"}, "journal claim retention differs")
        require(spec.get("updateStrategy", {}).get("type", "RollingUpdate") == "RollingUpdate" and spec.get("updateStrategy", {}).get("rollingUpdate", {}).get("partition") == 2, "journal partition differs")
        assert_claim(volume_claim_template(sts, "data"), "64Mi"); _token_policy(sts, False)
        init = _sole_container(sts, init=True); require("/data/seed" in command_text(init), "journal seed initializer differs")
        require(find_mount(init, "/data") is not None and find_mount(init, "/data").get("name") == "data", "journal initializer does not mount its claim")
        main = _sole_container(sts); _secret_input(sts, main, "KEY", "journal-key", "KEY")
        data_mount = find_mount(main, "/data")
        require(data_mount is not None and data_mount.get("name") == "data", "journal data claim mount differs")
        _has_source_mount(sts, main, "configMap", "journal-prefix")
        _port(main, "journal-http", 8080); _probe(main, "readinessProbe", "/", "journal-http", 5, None)
        return "three journal ordinals are Ready with partitioned retained state"

    def member_bodies() -> str:
        require(_endpoint_count(kube, "journal-svc") == 3, "journal Service does not have three Ready addresses")
        container = _sole_container(kube.get("statefulset", "journal"))["name"]
        for ordinal in range(3):
            require(_local_body(kube, f"journal-{ordinal}", container, 8080, f"journal-journal-{ordinal}-k1-seed\n"), f"journal-{ordinal} body differs")
        return "all three stable ordinals have exact bodies"

    def smoke() -> str:
        workload, container, result = _one_job(kube, "journal-smoke", 180, "journal-smoke-ok\n")
        require(pod_labels(workload).get("access") == "journal", "smoke access label differs")
        text = command_text(container)
        require(all(f"journal-{n}.journal-headless" in text for n in range(3)), "smoke omits a stable member")
        require(all(f"journal-journal-{n}-k1-seed" in text for n in range(3)), "smoke omits an exact member body")
        _bounded_attempts(text, 60)
        return result

    def partition_update() -> str:
        before = {n:kube.get("pod", f"journal-{n}")["metadata"]["uid"] for n in range(3)}
        old_annotations = kube.get("statefulset", "journal").get("spec", {}).get("template", {}).get("metadata", {}).get("annotations", {})
        old_value = old_annotations.get("experiment")
        patch_applied = False
        changed_uid: str | None = None
        try:
            kube.run(["patch","statefulset","journal","-n",kube.namespace,"--type=merge","-p",json.dumps({"spec":{"template":{"metadata":{"annotations":{"experiment":"v2"}}}}})])
            patch_applied = True
            kube.wait_until(lambda: kube.get("pod","journal-2")["metadata"]["uid"] != before[2] and _ready(kube.get("pod","journal-2")), "partitioned journal-2 update", 150, 2)
            changed_uid = kube.get("pod","journal-2")["metadata"]["uid"]
            require(all(kube.get("pod",f"journal-{n}")["metadata"]["uid"] == before[n] for n in (0,1)), "partition update replaced a protected ordinal")
        finally:
            if patch_applied:
                kube.run(["patch","statefulset","journal","-n",kube.namespace,"--type=merge","-p",json.dumps({"spec":{"template":{"metadata":{"annotations":{"experiment":old_value}}}}})])
                if changed_uid is not None:
                    kube.wait_until(lambda: kube.get("pod","journal-2")["metadata"]["uid"] != changed_uid and _ready(kube.get("pod","journal-2")), "partitioned journal-2 restoration", 150, 2)
        require(all(kube.get("pod",f"journal-{n}")["metadata"]["uid"] == before[n] for n in (0,1)), "restoration replaced a protected ordinal")
        return "only journal-2 rolled on change and restoration"

    def retained() -> str:
        container = _sole_container(kube.get("statefulset", "journal"))["name"]
        old = kube.get("pod","journal-1")["metadata"]["uid"]
        kube.exec("journal-1", "printf 'retained\\n' > /data/retained", container)
        try:
            kube.run(["delete","pod","journal-1","-n",kube.namespace,"--wait=true"])
            kube.wait_until(lambda: kube.get("pod","journal-1")["metadata"]["uid"] != old and _ready(kube.get("pod","journal-1")), "retained journal-1 replacement", 150, 2)
            kube.exec("journal-1", "test \"$(cat /data/seed)\" = seed && test \"$(cat /data/retained)\" = retained", container)
            kube.wait_until(lambda: _endpoint_count(kube,"journal-headless") == 3 and _endpoint_count(kube,"journal-svc") == 3, "journal endpoint recovery", 180, 2)
        finally:
            kube.run(["exec","-n",kube.namespace,"journal-1","-c",container,"--","rm","-f","/data/retained"], check=False)
        return "journal-1 retained seed and evaluator marker across replacement"

    return [
        ("R01","namespace and scope",lambda:_scope(kube,"hard-journal-ns",candidate_path)),
        ("R02","immutable journal prefix",lambda:_immutable_configmap(kube,"journal-prefix",{"PREFIX":"journal"})),
        ("R03","immutable journal key",lambda:_secret(kube,"journal-key","KEY","k1")),
        ("R04","headless identity Service",lambda:(_service(kube,"journal-headless",{"app":"journal"},8080,"journal-http",headless=True) and "headless journal Service is exact")),
        ("R05","partitioned retained StatefulSet",stateful),
        ("R06","member Service and ordinal bodies",lambda:(_service(kube,"journal-svc",{"app":"journal"},80,"journal-http") and member_bodies())),
        ("R07","bounded smoke Job",smoke),
        ("R08","pod hardening",lambda:_harden_with_writes(kube,[("statefulset","journal",{"/data"}),("job","journal-smoke",set())])),
        ("R09","network isolation",lambda:(_live_ingress_matrix(kube,[("aipc-eval-journal-ok","http://journal-svc",{"access":"journal"}),("aipc-eval-journal-headless-ok","http://journal-0.journal-headless:8080",{"access":"journal"})],[("aipc-eval-journal-deny","http://journal-svc"),("aipc-eval-journal-headless-deny","http://journal-0.journal-headless:8080")]) and _restricted_egress(kube,{"access":"journal"},[({"app":"journal"},{("TCP",8080),("TCP","journal-http")})]))),
        ("R10","availability budget",lambda:(_pdb(kube,"journal-pdb",{"app":"journal"},2) or "journal PDB is exact")),
        ("R11","resource governance",lambda:_limits(kube,"journal-defaults","journal-quota",{"pods":"10","persistentvolumeclaims":"4","requests.storage":"384Mi","services":"3","count/jobs.batch":"3","requests.cpu":"2","requests.memory":"1Gi","limits.cpu":"4","limits.memory":"2Gi"})),
        ("R12","initial convergence",member_bodies),
        ("R13","partitioned template update",partition_update),
        ("R14","retained member replacement",retained),
    ]


def _view_deployment(kube: Kubectl, name: str, component: str, port_name: str, port: int) -> str:
    dep = _workload_image_and_count(kube,"deployment",name,f"app={name}",2)
    _labels_include(dep, {"app":name,"component":component})
    container = _sole_container(dep); _emptydir_mount(dep,container,"/www")
    _port(container,port_name,port); _probe(container,"readinessProbe","/",port_name,5,2)
    text=command_text(container); require("fanout-source-svc" in text and "/www" in text and "sleep 2" in text and "rm" in text, f"{name} is not a failure-sensitive mirror")
    _bounded_fetch(text,5)
    return f"{name} has two Ready failure-sensitive mirrors"


def _hard13_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def source() -> str:
        dep=_workload_image_and_count(kube,"deployment","fanout-source","app=fanout-source",2)
        _labels_include(dep, {"app":"fanout-source","component":"source"})
        _rolling_controls(dep,replicas=2,min_ready=5)
        _token_policy(dep,False); c=_sole_container(dep); _secret_input(dep,c,"TOKEN","fanout-key","TOKEN")
        _mount_source(dep,c,"/srv","configMap","fanout-page",read_only=True); _port(c,"source-http",8080); _probe(c,"readinessProbe","/","source-http",5,None)
        require("fanout-ok" in command_text(c) or "TOKEN" in command_text(c),"source does not verify its token")
        return "two source replicas are Ready with exact projected inputs"

    def update() -> str:
        return _update_configmap(kube,"fanout-page","index.html","source=v2\n","source=v1\n",lambda v:_all_pod_bodies(kube,"app=fanout-source",_sole_container(kube.get("deployment","fanout-source"))["name"],8080,v,2) and _all_pod_bodies(kube,"app=fanout-alpha",_sole_container(kube.get("deployment","fanout-alpha"))["name"],8081,f"alpha:{v}",2) and _all_pod_bodies(kube,"app=fanout-beta",_sole_container(kube.get("deployment","fanout-beta"))["name"],8082,f"beta:{v}",2),180,[("deployment","fanout-source"),("deployment","fanout-alpha"),("deployment","fanout-beta")],["app=fanout-source","app=fanout-alpha","app=fanout-beta"])

    def outage() -> str:
        result = _service_outage(kube,"fanout-source-svc",lambda:_ready_count(kube,"app=fanout-alpha")==0 and _ready_count(kube,"app=fanout-beta")==0 and _endpoint_count(kube,"fanout-alpha-svc")==0 and _endpoint_count(kube,"fanout-beta-svc")==0,lambda:_endpoint_count(kube,"fanout-source-svc")==2 and _endpoint_count(kube,"fanout-alpha-svc")==2 and _endpoint_count(kube,"fanout-beta-svc")==2,90,180)
        _body_probe(kube,"aipc-eval-alpha-restored","http://fanout-alpha-svc","alpha:source=v1\n",labels={"access":"fanout"})
        _body_probe(kube,"aipc-eval-beta-restored","http://fanout-beta-svc","beta:source=v1\n",labels={"access":"fanout"})
        return result + "; both exact view bodies recovered"

    def alpha_replacement() -> str:
        return _replacement_with_stable_peer(kube,"app=fanout-alpha","fanout-alpha-svc","alpha:source=v1\n","app=fanout-beta","fanout-beta-svc","beta:source=v1\n",{"access":"fanout"},90)

    return [
        ("R01","namespace and scope",lambda:_scope(kube,"hard-fanout-ns",candidate_path)),
        ("R02","mutable source page",lambda:_exact_configmap(kube,"fanout-page",{"index.html":"source=v1\n"})),
        ("R03","immutable fanout key",lambda:_secret(kube,"fanout-key","TOKEN","fanout-ok")),
        ("R04","source Deployment",source),
        ("R05","source Service",lambda:(_service(kube,"fanout-source-svc",{"app":"fanout-source"},8080,"source-http") and _body_probe(kube,"aipc-eval-fanout-source","http://fanout-source-svc:8080","source=v1\n",labels={"access":"fanout"}))),
        ("R06","alpha mirror",lambda:(_view_deployment(kube,"fanout-alpha","alpha","alpha-http",8081) and _service(kube,"fanout-alpha-svc",{"app":"fanout-alpha"},80,"alpha-http") and _body_probe(kube,"aipc-eval-alpha","http://fanout-alpha-svc","alpha:source=v1\n",labels={"access":"fanout"}))),
        ("R07","beta mirror",lambda:(_view_deployment(kube,"fanout-beta","beta","beta-http",8082) and _service(kube,"fanout-beta-svc",{"app":"fanout-beta"},80,"beta-http") and _body_probe(kube,"aipc-eval-beta","http://fanout-beta-svc","beta:source=v1\n",labels={"access":"fanout"}))),
        ("R08","pod hardening",lambda:_harden_with_writes(kube,[("deployment","fanout-source",set()),("deployment","fanout-alpha",{"/www"}),("deployment","fanout-beta",{"/www"})])),
        ("R09","network isolation",lambda:(_live_ingress_matrix(kube,[("aipc-eval-source-alpha","http://fanout-source-svc:8080",{"component":"alpha"}),("aipc-eval-source-beta","http://fanout-source-svc:8080",{"component":"beta"}),("aipc-eval-source-access","http://fanout-source-svc:8080",{"access":"fanout"}),("aipc-eval-alpha-access","http://fanout-alpha-svc",{"access":"fanout"}),("aipc-eval-beta-access","http://fanout-beta-svc",{"access":"fanout"})],[("aipc-eval-source-deny","http://fanout-source-svc:8080"),("aipc-eval-alpha-deny","http://fanout-alpha-svc"),("aipc-eval-beta-deny","http://fanout-beta-svc")]) and _restricted_egress(kube,{"app":"fanout-source","component":"source"},[],dns=False) and _restricted_egress(kube,{"app":"fanout-alpha","component":"alpha"},[({"app":"fanout-source","component":"source"},{("TCP",8080),("TCP","source-http")})]) and _restricted_egress(kube,{"app":"fanout-beta","component":"beta"},[({"app":"fanout-source","component":"source"},{("TCP",8080),("TCP","source-http")})]))),
        ("R10","availability budgets",lambda:_pdbs_by_app(kube,{"fanout-source":1,"fanout-alpha":1,"fanout-beta":1})),
        ("R11","resource governance",lambda:_limits(kube,"fanout-defaults","fanout-quota",{"pods":"14","services":"5","requests.cpu":"2","requests.memory":"1Gi","limits.cpu":"4","limits.memory":"2Gi"})),
        ("R12","fanout live propagation",update),
        ("R13","source Service outage",outage),
        ("R14","isolated alpha replacement",alpha_replacement),
    ]


def _input_deployment(kube: Kubectl, name: str, component: str, key: str, port_name: str, port: int) -> str:
    dep=_workload_image_and_count(kube,"deployment",name,f"app={name}",2)
    _labels_include(dep, {"app":name,"component":component})
    _token_policy(dep,False); c=_sole_container(dep)
    _single_config_key_projection(dep, c, "quorum-inputs", key)
    _port(c,port_name,port); _probe(c,"readinessProbe",None,port_name,None,None)
    return f"{name} has two Ready replicas with only the {key} input"


def _hard14_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def quorum() -> str:
        dep=_workload_image_and_count(kube,"deployment","quorum-view","app=quorum-view",2)
        _labels_include(dep, {"app":"quorum-view","component":"quorum"})
        _rolling_controls(dep,replicas=2,min_ready=5); _token_policy(dep,False)
        c=_sole_container(dep); _secret_input(dep,c,"TOKEN","quorum-key","TOKEN"); _emptydir_mount(dep,c,"/www")
        _port(c,"quorum-http",8082); _probe(c,"readinessProbe","/","quorum-http",5,2)
        text=command_text(c); require("quorum-alpha-svc" in text and "quorum-beta-svc" in text and "TOKEN" in text and "/www/index.html" in text and "rm" in text and "sleep 2" in text,"quorum does not require both live inputs")
        _bounded_fetch(text,None)
        return "two Ready quorum pods require both bounded input requests"

    def update() -> str:
        def bodies(value: str) -> bool:
            view_value = "quorum=a2+b1\n" if value == "a2\n" else "quorum=a1+b1\n"
            return (
                _all_pod_bodies(
                    kube,
                    "app=quorum-alpha",
                    _sole_container(kube.get("deployment", "quorum-alpha"))["name"],
                    8080,
                    value,
                    2,
                )
                and _all_pod_bodies(
                    kube,
                    "app=quorum-view",
                    _sole_container(kube.get("deployment", "quorum-view"))["name"],
                    8082,
                    view_value,
                    2,
                )
            )

        return _update_configmap(
            kube,
            "quorum-inputs",
            "alpha",
            "a2\n",
            "a1\n",
            bodies,
            150,
            [("deployment", "quorum-alpha"), ("deployment", "quorum-beta"), ("deployment", "quorum-view")],
            ["app=quorum-alpha", "app=quorum-beta", "app=quorum-view"],
        )

    def beta_outage() -> str:
        result = _service_outage(kube,"quorum-beta-svc",lambda:_ready_count(kube,"app=quorum-view")==0 and _endpoint_count(kube,"quorum-svc")==0 and _ready_count(kube,"app=quorum-alpha")==2,lambda:_endpoint_count(kube,"quorum-alpha-svc")==2 and _endpoint_count(kube,"quorum-beta-svc")==2 and _endpoint_count(kube,"quorum-svc")==2,90,180)
        _body_probe(kube,"aipc-eval-quorum-restored","http://quorum-svc","quorum=a1+b1\n",labels={"access":"quorum"})
        return result + "; exact quorum body recovered"

    return [
        ("R01","namespace and scope",lambda:_scope(kube,"hard-quorum-ns",candidate_path)),
        ("R02","mutable inputs and alpha propagation",lambda:(_exact_configmap(kube,"quorum-inputs",{"alpha":"a1\n","beta":"b1\n"}) and update())),
        ("R03","immutable quorum key",lambda:_secret(kube,"quorum-key","TOKEN","quorum-ok")),
        ("R04","alpha input Deployment",lambda:_input_deployment(kube,"quorum-alpha","alpha","alpha","alpha-http",8080)),
        ("R05","alpha Service",lambda:(_service(kube,"quorum-alpha-svc",{"app":"quorum-alpha"},8080,"alpha-http") and _body_probe(kube,"aipc-eval-quorum-alpha","http://quorum-alpha-svc:8080","a1\n",labels={"access":"quorum"}))),
        ("R06","beta input Deployment",lambda:_input_deployment(kube,"quorum-beta","beta","beta","beta-http",8081)),
        ("R07","beta Service",lambda:(_service(kube,"quorum-beta-svc",{"app":"quorum-beta"},8081,"beta-http") and _body_probe(kube,"aipc-eval-quorum-beta","http://quorum-beta-svc:8081","b1\n",labels={"access":"quorum"}))),
        ("R08","dual-gated quorum Deployment",quorum),
        ("R09","quorum Service continuity",lambda:(_service(kube,"quorum-svc",{"app":"quorum-view"},80,"quorum-http") and _body_probe(kube,"aipc-eval-quorum","http://quorum-svc","quorum=a1+b1\n",labels={"access":"quorum"}) and _replacement_continuity(kube,"app=quorum-view","quorum-svc","http://quorum-svc","quorum=a1+b1\n",{"access":"quorum"},90))),
        ("R10","pod hardening",lambda:_harden_with_writes(kube,[("deployment","quorum-alpha",set()),("deployment","quorum-beta",set()),("deployment","quorum-view",{"/www"})])),
        ("R11","network isolation",lambda:(_live_ingress_matrix(kube,[("aipc-eval-alpha-quorum","http://quorum-alpha-svc:8080",{"component":"quorum"}),("aipc-eval-alpha-access","http://quorum-alpha-svc:8080",{"access":"quorum"}),("aipc-eval-beta-quorum","http://quorum-beta-svc:8081",{"component":"quorum"}),("aipc-eval-beta-access","http://quorum-beta-svc:8081",{"access":"quorum"}),("aipc-eval-quorum-access","http://quorum-svc",{"access":"quorum"})],[("aipc-eval-alpha-deny","http://quorum-alpha-svc:8080"),("aipc-eval-beta-deny","http://quorum-beta-svc:8081"),("aipc-eval-quorum-deny","http://quorum-svc")]) and _restricted_egress(kube,{"app":"quorum-alpha","component":"alpha"},[],dns=False) and _restricted_egress(kube,{"app":"quorum-beta","component":"beta"},[],dns=False) and _restricted_egress(kube,{"app":"quorum-view","component":"quorum"},[({"app":"quorum-alpha","component":"alpha"},{("TCP",8080),("TCP","alpha-http")}),({"app":"quorum-beta","component":"beta"},{("TCP",8081),("TCP","beta-http")})]))),
        ("R12","availability budgets",lambda:_pdbs_by_app(kube,{"quorum-alpha":1,"quorum-beta":1,"quorum-view":1})),
        ("R13","resource governance",lambda:_limits(kube,"quorum-defaults","quorum-quota",{"pods":"14","services":"5","requests.cpu":"2","requests.memory":"1Gi","limits.cpu":"4","limits.memory":"2Gi"})),
        ("R14","beta Service outage",beta_outage),
    ]


def _hard15_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def server() -> str:
        dep=_workload_image_and_count(kube,"deployment","handoff-server","app=handoff-server",1); _recreate(dep); _token_policy(dep,False)
        init=_sole_container(dep,init=True); require("/data/base" in command_text(init),"handoff initializer differs")
        _mount_source(dep,init,"/data","persistentVolumeClaim","handoff-data")
        c=_sole_container(dep); _secret_input(dep,c,"KEY","handoff-key","KEY")
        _has_source_mount(dep,c,"configMap","handoff-prefix"); _mount_source(dep,c,"/data","persistentVolumeClaim","handoff-data")
        _port(c,"handoff-http",8080); _probe(c,"readinessProbe","/","handoff-http",5,None)
        require("/data/index.html" in command_text(c) and "/data/base" in command_text(c) and "KEY" in command_text(c),"handoff body does not derive from all declared inputs")
        return "one Ready Recreate server uses the retained claim and projected inputs"

    def maintenance_spec() -> tuple[dict[str, Any], dict[str, Any]]:
        item,container=_cron_structure(kube,"handoff-maintain","23 */2 * * *",150)
        workload={"spec":{"template":item["spec"]["jobTemplate"]["spec"]["template"]}}
        require(pod_labels(workload).get("role")=="handoff-maintenance","maintenance role label differs")
        _mount_source(workload,container,"/data","persistentVolumeClaim","handoff-data")
        require(container.get("image")=="busybox:1.36.1","maintenance image differs")
        text=command_text(container); require("handoff-svc" in text and "/data/maintained" in text and "handoff-seal-base" in text,"maintenance contract differs")
        _bounded_attempts(text,60)
        return item,container

    def cron_spec_and_run() -> str:
        maintenance_spec()
        return _cron_job(kube,"handoff-maintain","aipc-eval-handoff-maintain",150,"handoff-maintain-ok\n")

    def update() -> str:
        return _update_configmap(kube,"handoff-prefix","PREFIX","handoff2","handoff",lambda v:_all_pod_bodies(kube,"app=handoff-server",_sole_container(kube.get("deployment","handoff-server"))["name"],8080,f"{v}-seal-base\n",1),150,[("deployment","handoff-server")],["app=handoff-server"])

    def replacement() -> str:
        pod=next(p for p in kube.list("pods","app=handoff-server") if _ready(p)); old=pod["metadata"]["uid"]
        c=_sole_container(kube.get("deployment","handoff-server"))["name"]
        kube.exec(pod["metadata"]["name"],"printf 'survivor\\n' > /data/survivor",c)
        try:
            kube.run(["delete","pod",pod["metadata"]["name"],"-n",kube.namespace,"--wait=true"])
            kube.wait_until(lambda:any(_ready(p) and p["metadata"]["uid"]!=old for p in kube.list("pods","app=handoff-server")),"different-UID handoff replacement",150,2)
            new=next(p for p in kube.list("pods","app=handoff-server") if _ready(p) and p["metadata"]["uid"]!=old)
            kube.exec(new["metadata"]["name"],"test \"$(cat /data/base)\" = base && test \"$(cat /data/survivor)\" = survivor",c)
            kube.wait_until(lambda:_endpoint_count(kube,"handoff-svc")==1,"handoff endpoint recovery",180,2)
            _body_probe(kube,"aipc-eval-handoff-recovery","http://handoff-svc","handoff-seal-base\n",labels={"access":"handoff"})
        finally:
            for item in kube.list("pods","app=handoff-server"):
                kube.run(["exec","-n",kube.namespace,item["metadata"]["name"],"-c",c,"--","rm","-f","/data/survivor"],check=False)
        return "base and evaluator marker survived a different-UID replacement"

    def idempotent_cron() -> str:
        first=cron_spec_and_run()
        pod=next(p for p in kube.list("pods","app=handoff-server") if _ready(p)); c=_sole_container(kube.get("deployment","handoff-server"))["name"]
        kube.exec(pod["metadata"]["name"],"test \"$(cat /data/maintained)\" = maintained",c)
        before=kube.exec(pod["metadata"]["name"],"sha256sum /data/maintained | cut -d' ' -f1",c).strip()
        second=cron_spec_and_run()
        after=kube.exec(pod["metadata"]["name"],"sha256sum /data/maintained | cut -d' ' -f1",c).strip()
        require(before==after,"second maintenance run changed the marker")
        return first+"; "+second+"; marker remained exact"

    return [
        ("R01","namespace and scope",lambda:_scope(kube,"hard-handoff-ns",candidate_path)),
        ("R02","mutable handoff prefix",lambda:_exact_configmap(kube,"handoff-prefix",{"PREFIX":"handoff"})),
        ("R03","immutable handoff key",lambda:_secret(kube,"handoff-key","KEY","seal")),
        ("R04","durable claim",lambda:_claim(kube,"handoff-data","96Mi")),
        ("R05","retained handoff server",server),
        ("R06","handoff Service",lambda:(_service(kube,"handoff-svc",{"app":"handoff-server"},80,"handoff-http") and _body_probe(kube,"aipc-eval-handoff","http://handoff-svc","handoff-seal-base\n",labels={"access":"handoff"}))),
        ("R07","maintenance CronJob contract",lambda: (maintenance_spec() and "maintenance CronJob structure is exact")),
        ("R08","pod hardening",lambda:_harden_with_writes(kube,[("deployment","handoff-server",{"/data"}),("cronjob","handoff-maintain",{"/data"})])),
        ("R09","network isolation",lambda:(_live_ingress_matrix(kube,[("aipc-eval-handoff-access","http://handoff-svc",{"access":"handoff"}),("aipc-eval-handoff-maint","http://handoff-svc",{"role":"handoff-maintenance"})],[("aipc-eval-handoff-deny","http://handoff-svc")]) and _restricted_egress(kube,{"role":"handoff-maintenance"},[({"app":"handoff-server"},{("TCP",8080),("TCP","handoff-http")})]))),
        ("R10","availability budget",lambda:(_pdb(kube,"handoff-pdb",{"app":"handoff-server"},1) or "handoff PDB is exact")),
        ("R11","resource governance",lambda:_limits(kube,"handoff-defaults","handoff-quota",{"pods":"8","persistentvolumeclaims":"2","requests.storage":"256Mi","services":"2","count/jobs.batch":"3","count/cronjobs.batch":"2","requests.cpu":"1","requests.memory":"512Mi","limits.cpu":"2","limits.memory":"1Gi"})),
        ("R12","live prefix projection",update),
        ("R13","durable server replacement",replacement),
        ("R14","idempotent maintenance executions",idempotent_cron),
    ]


PENDING_FACTORIES: dict[str, Callable[[Kubectl, Path | None], list[Check]]] = {
    "hard-006": _hard6_checks,
    "hard-007": _hard7_checks,
    "hard-008": _hard8_checks,
    "hard-009": _hard9_checks,
    "hard-010": _hard10_checks,
    "hard-011": _hard11_checks,
    "hard-012": _hard12_checks,
    "hard-013": _hard13_checks,
    "hard-014": _hard14_checks,
    "hard-015": _hard15_checks,
}


def pending_checks(task_id: str, kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    try:
        checks = PENDING_FACTORIES[task_id](kube, candidate_path)
    except KeyError as exc:
        raise EvaluationInfrastructureError(f"pending oracle is not implemented for {task_id}") from exc
    ids = [item[0] for item in checks]
    require(set(ids) == PENDING_EXPECTED_REQUIREMENTS[task_id], "pending suite coverage is incomplete")
    require(len(ids) == len(set(ids)), "pending suite has duplicate requirement checks")
    return checks
