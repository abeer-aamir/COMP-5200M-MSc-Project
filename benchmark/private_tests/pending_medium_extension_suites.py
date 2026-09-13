"""Hidden oracles for medium-006 through medium-015.

The historical ``pending_`` filename is retained for import stability, but the
module is registered in the live dispatcher. Checks score the public contract: harmless
extra labels, arbitrary container names, numeric or named Service targets, split or
merged NetworkPolicies, and equivalent bounded shell programs are accepted.  Exact
values are asserted only when the public task states them.  Every evaluator mutation
has an unconditional restoration path.
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
    job_terminal_condition,
    pod_labels,
    pod_spec,
    require,
    volume_claim_template,
)
from .very_hard_suites import _scope


Check = tuple[str, str, Callable[[], str | None]]

PENDING_EXPECTED_REQUIREMENTS = {
    task_id: {f"R{number:02d}" for number in range(1, count + 1)}
    for task_id, count in {
        "medium-006": 10,
        "medium-007": 10,
        "medium-008": 9,
        "medium-009": 10,
        "medium-010": 10,
        "medium-011": 10,
        "medium-012": 10,
        "medium-013": 11,
        "medium-014": 11,
        "medium-015": 10,
    }.items()
}


def _sole_container(workload: dict[str, Any], *, init: bool = False) -> dict[str, Any]:
    key = "initContainers" if init else "containers"
    containers = pod_spec(workload).get(key, [])
    require(len(containers) == 1, f"expected exactly one {key[:-1]}")
    return containers[0]


def _job_workload(job: dict[str, Any]) -> dict[str, Any]:
    return {"spec": {"template": job["spec"]["template"]}}


def _cron_workload(cronjob: dict[str, Any]) -> dict[str, Any]:
    return {
        "spec": {
            "template": cronjob["spec"]["jobTemplate"]["spec"]["template"]
        }
    }


def _labels_include(workload: dict[str, Any], expected: dict[str, str]) -> None:
    labels = pod_labels(workload)
    for key, value in expected.items():
        require(labels.get(key) == value, f"pod label {key}={value} is missing")


def _exact_configmap(
    kube: Kubectl,
    name: str,
    expected: dict[str, str],
    *,
    immutable: bool,
) -> str:
    item = kube.get("configmap", name)
    require(item.get("data", {}) == expected, f"{name} data differs")
    require(not item.get("binaryData"), f"{name} has unexpected binaryData")
    if immutable:
        require(item.get("immutable") is True, f"{name} must be immutable")
    else:
        require(item.get("immutable") is not True, f"{name} must remain mutable")
    return f"{name} has the exact {'immutable' if immutable else 'mutable'} data contract"


def _exact_secret(
    kube: Kubectl,
    name: str,
    key: str,
    value: str,
    *,
    immutable: bool,
) -> str:
    item = kube.get("secret", name)
    require(item.get("type") == "Opaque", f"{name} must be Opaque")
    require(set(item.get("data", {})) == {key}, f"{name} key set differs")
    try:
        decoded = base64.b64decode(item["data"][key], validate=True).decode("utf-8")
    except Exception as exc:
        raise RequirementFailure(f"{name}/{key} is not valid base64 UTF-8") from exc
    require(decoded == value, f"{name}/{key} differs")
    if immutable:
        require(item.get("immutable") is True, f"{name} must be immutable")
    else:
        require(item.get("immutable") is not True, f"{name} must remain mutable")
    return f"{name} is exact and {'immutable' if immutable else 'mutable'}"


def _source_name(volume: dict[str, Any], kind: str) -> str | None:
    direct = volume.get(kind, {})
    if kind == "secret":
        name = direct.get("secretName")
    elif kind == "persistentVolumeClaim":
        name = direct.get("claimName")
    else:
        name = direct.get("name")
    if name:
        return str(name)
    for item in volume.get("projected", {}).get("sources", []) or []:
        source = item.get(kind, {}) if isinstance(item, dict) else {}
        candidate = source.get("secretName" if kind == "secret" else "name")
        if candidate:
            return str(candidate)
    return None


def _mount_source(
    workload: dict[str, Any],
    container: dict[str, Any],
    path: str,
    kind: str,
    name: str,
    *,
    key_path: tuple[str, str] | None = None,
) -> None:
    mount = find_mount(container, path)
    require(mount is not None, f"mount {path} is absent")
    volume = find_volume(pod_spec(workload), mount.get("name", ""))
    require(volume is not None, f"volume for {path} is absent")
    require(_source_name(volume, kind) == name, f"source for {path} differs")
    if key_path is None:
        return
    key, projected_path = key_path
    source = volume.get(kind, {})
    if not source:
        sources = [
            item.get(kind, {})
            for item in volume.get("projected", {}).get("sources", []) or []
            if isinstance(item, dict) and kind in item
        ]
        require(len(sources) == 1, f"projection for {path} is ambiguous")
        source = sources[0]
    require(
        any(
            isinstance(item, dict)
            and item.get("key") == key
            and item.get("path") == projected_path
            for item in source.get("items", []) or []
        ),
        f"{name}/{key} is not projected as {projected_path}",
    )


def _emptydir(workload: dict[str, Any], container: dict[str, Any], path: str) -> None:
    mount = find_mount(container, path)
    require(mount is not None, f"{path} mount is absent")
    volume = find_volume(pod_spec(workload), mount.get("name", ""))
    require(volume is not None and "emptyDir" in volume, f"{path} is not an emptyDir")


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
        f"{env} Secret reference differs",
    )


def _config_env(container: dict[str, Any], env: str, configmap: str, key: str) -> None:
    matches = [
        item
        for item in container.get("env", []) or []
        if isinstance(item, dict) and item.get("name") == env
    ]
    require(len(matches) == 1, f"environment variable {env} must be declared once")
    reference = matches[0].get("valueFrom", {}).get("configMapKeyRef", {})
    require(
        reference.get("name") == configmap and reference.get("key") == key,
        f"{env} ConfigMap reference differs",
    )


def _projected_input(
    workload: dict[str, Any],
    container: dict[str, Any],
    env: str,
    source_kind: str,
    source_name: str,
    key: str,
) -> None:
    """Accept an environment reference or an equivalent mounted key."""

    reference_key = "secretKeyRef" if source_kind == "secret" else "configMapKeyRef"
    for item in container.get("env", []) or []:
        if not isinstance(item, dict) or item.get("name") != env:
            continue
        reference = item.get("valueFrom", {}).get(reference_key, {})
        if reference.get("name") == source_name and reference.get("key") == key:
            require(
                reference.get("optional", False) is False,
                f"{env} must not make {source_kind} {source_name} optional",
            )
            return

    spec = pod_spec(workload)
    program = command_text(container)
    for mount in container.get("volumeMounts", []) or []:
        volume = find_volume(spec, mount.get("name", ""))
        if volume is None or _source_name(volume, source_kind) != source_name:
            continue
        source = volume.get(source_kind, {})
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
    raise RequirementFailure(
        f"{env} is not sourced from {source_kind} {source_name}/{key}"
    )


def _port(container: dict[str, Any], name: str, number: int) -> None:
    matches = [
        item
        for item in container.get("ports", []) or []
        if item.get("name") == name
        and item.get("containerPort") == number
        and item.get("protocol", "TCP") == "TCP"
    ]
    require(len(matches) == 1, f"named TCP port {name}:{number} differs")


def _http_readiness(
    container: dict[str, Any],
    port_name: str,
    port_number: int,
    *,
    period: int,
    failure: int | None,
) -> None:
    probe = container.get("readinessProbe", {})
    target = probe.get("httpGet", {})
    require(target.get("path") == "/", "readiness path must be /")
    require(
        int_or_string(target.get("port")) in {port_name, str(port_number)},
        "readiness port differs",
    )
    require(probe.get("periodSeconds", 10) == period, "readiness period differs")
    if failure is not None:
        require(probe.get("failureThreshold", 3) == failure, "readiness failure threshold differs")


def _token_policy(workload: dict[str, Any], expected: bool) -> None:
    require(
        pod_spec(workload).get("automountServiceAccountToken") is expected,
        "automountServiceAccountToken differs",
    )


def _rolling(
    workload: dict[str, Any],
    *,
    replicas: int,
    min_ready: int,
    deadline: int,
    history: int | None = None,
) -> None:
    spec = workload.get("spec", {})
    require(spec.get("replicas") == replicas, "replica count differs")
    strategy = spec.get("strategy", {})
    require(strategy.get("type", "RollingUpdate") == "RollingUpdate", "strategy is not RollingUpdate")
    rolling = strategy.get("rollingUpdate", {})
    require(int_or_string(rolling.get("maxUnavailable")) == "0", "maxUnavailable must be 0")
    require(int_or_string(rolling.get("maxSurge")) == "1", "maxSurge must be 1")
    require(spec.get("minReadySeconds") == min_ready, "minReadySeconds differs")
    require(spec.get("progressDeadlineSeconds") == deadline, "progressDeadlineSeconds differs")
    if history is not None:
        require(spec.get("revisionHistoryLimit") == history, "revisionHistoryLimit differs")


def _ready(pod: dict[str, Any]) -> bool:
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in pod.get("status", {}).get("conditions", []) or []
    )


def _ready_pods(kube: Kubectl, selector: str) -> list[dict[str, Any]]:
    return [pod for pod in kube.list("pods", selector) if _ready(pod)]


def _ready_count(kube: Kubectl, selector: str) -> int:
    return len(_ready_pods(kube, selector))


def _endpoint_uids(kube: Kubectl, service: str) -> set[str]:
    return {
        address.get("targetRef", {}).get("uid", "")
        for subset in kube.get("endpoints", service).get("subsets", []) or []
        for address in subset.get("addresses", []) or []
        if address.get("targetRef", {}).get("uid")
    }


def _endpoint_count(kube: Kubectl, service: str) -> int:
    return len(_endpoint_uids(kube, service))


def _service(
    kube: Kubectl,
    name: str,
    selector: dict[str, str],
    port: int,
    target_name: str,
    target_number: int,
    *,
    headless: bool = False,
) -> dict[str, Any]:
    item = kube.get("service", name)
    spec = item.get("spec", {})
    require(spec.get("type", "ClusterIP") == "ClusterIP", f"{name} is not ClusterIP")
    actual_selector = spec.get("selector", {})
    for key, value in selector.items():
        require(actual_selector.get(key) == value, f"{name} selector misses {key}={value}")
    if headless:
        require(spec.get("clusterIP") == "None", f"{name} is not headless")
        require(spec.get("publishNotReadyAddresses", False) is False, f"{name} publishes NotReady addresses")
    else:
        require(spec.get("clusterIP") not in {None, "", "None"}, f"{name} lacks a cluster IP")
    ports = [candidate for candidate in spec.get("ports", []) or [] if candidate.get("port") == port]
    require(len(ports) == 1, f"{name} must expose the required port once")
    exposed = ports[0]
    require(exposed.get("protocol", "TCP") == "TCP", f"{name} port is not TCP")
    require(
        int_or_string(exposed.get("targetPort", port)) in {target_name, str(target_number)},
        f"{name} targetPort differs",
    )
    return item


def _body_probe(
    kube: Kubectl,
    name: str,
    url: str,
    expected: str,
    *,
    labels: dict[str, str],
) -> str:
    encoded = base64.b64encode(expected.encode()).decode()
    return kube.run_probe_pod(
        name,
        f'test "$(wget -q -T 5 -O - {url} | base64 | tr -d \'\\n\')" = {encoded}',
        labels=labels,
        timeout_seconds=35,
    )


def _local_body(
    kube: Kubectl,
    pod: str,
    container: str,
    port: int,
    expected: str,
) -> bool:
    encoded = base64.b64encode(expected.encode()).decode()
    result = kube.run(
        [
            "exec", "-n", kube.namespace, pod, "-c", container, "--", "sh", "-ec",
            f'test "$(wget -q -T 5 -O - http://127.0.0.1:{port}/ | base64 | tr -d \'\\n\')" = {encoded}',
        ],
        check=False,
    )
    return result.returncode == 0


def _all_pod_bodies(
    kube: Kubectl,
    selector: str,
    container: str,
    port: int,
    expected: Callable[[str], str] | str,
    count: int,
) -> bool:
    pods = _ready_pods(kube, selector)
    if len(pods) != count:
        return False
    for pod in pods:
        name = pod.get("metadata", {}).get("name", "")
        body = expected(name) if callable(expected) else expected
        if not _local_body(kube, name, container, port, body):
            return False
    return True


def _writable_data_paths(workload: dict[str, Any]) -> set[str]:
    spec = pod_spec(workload)
    claims = {
        item.get("metadata", {}).get("name")
        for item in workload.get("spec", {}).get("volumeClaimTemplates", []) or []
    }
    paths: set[str] = set()
    for container in all_containers(spec):
        for mount in container.get("volumeMounts", []) or []:
            if mount.get("readOnly") is True:
                continue
            volume = find_volume(spec, mount.get("name", ""))
            data_backed = mount.get("name") in claims or (
                volume is not None
                and any(key in volume for key in ("emptyDir", "persistentVolumeClaim", "hostPath"))
            )
            if data_backed:
                paths.add(str(mount.get("mountPath", "")))
    return paths


def _hardened(workload: dict[str, Any], label: str, writable: set[str]) -> str:
    spec = pod_spec(workload)
    pod_security = spec.get("securityContext", {})
    require(pod_security.get("fsGroup") == 1000, f"{label} fsGroup differs")
    for container in all_containers(spec):
        context = container.get("securityContext", {})
        user = context.get("runAsUser", pod_security.get("runAsUser"))
        group = context.get("runAsGroup", pod_security.get("runAsGroup"))
        non_root = context.get("runAsNonRoot", pod_security.get("runAsNonRoot"))
        require(user == 1000 and group == 1000, f"{label} UID/GID differs")
        require(non_root is not False, f"{label} explicitly permits root")
        require(context.get("allowPrivilegeEscalation") is False, f"{label} permits privilege escalation")
        require(context.get("readOnlyRootFilesystem") is True, f"{label} root filesystem is writable")
        require("ALL" in set(context.get("capabilities", {}).get("drop", [])), f"{label} does not drop ALL capabilities")
        require(not context.get("capabilities", {}).get("add"), f"{label} adds capabilities after dropping ALL")
    require(_writable_data_paths(workload) == writable, f"{label} writable data paths differ")
    return f"{label} is hardened with only the declared writable data paths"


def _pdb(kube: Kubectl, name: str, selector: dict[str, str], minimum: int) -> str:
    item = kube.get("pdb", name)
    spec = item.get("spec", {})
    actual = spec.get("selector", {}).get("matchLabels", {})
    for key, value in selector.items():
        require(actual.get(key) == value, f"{name} selector misses {key}={value}")
    for pod in kube.list("pods", ",".join(f"{key}={value}" for key, value in selector.items())):
        labels = pod.get("metadata", {}).get("labels", {})
        require(all(labels.get(key) == value for key, value in actual.items()), f"{name} excludes a required pod")
    require(int_or_string(spec.get("minAvailable")) == str(minimum), f"{name} minAvailable differs")
    require("maxUnavailable" not in spec, f"{name} must not set maxUnavailable")
    return f"{name} selects all required pods with minAvailable {minimum}"


def _limit_range(kube: Kubectl, name: str) -> str:
    entries = kube.get("limitrange", name).get("spec", {}).get("limits", [])
    containers = [entry for entry in entries if entry.get("type") == "Container"]
    require(len(containers) == 1, f"{name} must contain one Container limit")
    entry = containers[0]
    require(entry.get("defaultRequest") == {"cpu": "20m", "memory": "32Mi"}, f"{name} default requests differ")
    require(entry.get("default") == {"cpu": "200m", "memory": "128Mi"}, f"{name} default limits differ")
    return f"{name} has the declared container defaults"


def _quota(kube: Kubectl, name: str, expected: dict[str, str]) -> str:
    hard = kube.get("resourcequota", name).get("spec", {}).get("hard", {})
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
            f"{name} declares conflicting quota aliases for {normalized}",
        )
        actual[normalized] = rendered
    require(actual == expected, f"{name} hard limits differ")
    return f"{name} has the exact hard limits"


def _resource_contract(container: dict[str, Any]) -> None:
    resources = container.get("resources", {})
    require(resources.get("requests") == {"cpu": "20m", "memory": "32Mi"}, "container requests differ")
    require(resources.get("limits") == {"cpu": "200m", "memory": "128Mi"}, "container limits differ")


def _live_ingress(
    kube: Kubectl,
    allowed: list[tuple[str, str, dict[str, str]]],
    denied: list[tuple[str, str]],
) -> str:
    require(kube.list("networkpolicies"), "no NetworkPolicy was created")
    for name, url, labels in allowed:
        kube.run_probe_pod(name, f"wget -q -T 5 -O /dev/null {url}", labels=labels)
    for name, url in denied:
        kube.run_probe_pod(
            name,
            f"wget -q -T 4 -O /dev/null {url}",
            labels={},
            expect_success=False,
            timeout_seconds=20,
        )
    return "the declared positive and negative ingress paths were observed live"


def _generations(kube: Kubectl, resources: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    return {
        (kind, name): kube.get(kind, name).get("metadata", {}).get("generation", 0)
        for kind, name in resources
    }


def _uids(kube: Kubectl, selector: str) -> set[str]:
    return {
        pod.get("metadata", {}).get("uid", "")
        for pod in kube.list("pods", selector)
        if pod.get("metadata", {}).get("uid")
    }


def _candidate_mutation_failure(kube: Kubectl, exc: KubectlError, description: str) -> None:
    detail = f"{exc.stdout}\n{exc.stderr}".lower()
    markers = ("immutable", "forbidden", "denied", "invalid", "not found", "notfound")
    if not any(marker in detail for marker in markers):
        raise exc
    ready = kube.run(["get", "--raw=/readyz"], check=False, timeout=min(kube.command_timeout, 10))
    if ready.returncode == 0 and "ok" in ready.stdout.lower():
        raise RequirementFailure(
            f"the evaluator could not perform {description} because the candidate rejected it while the API remained healthy"
        ) from exc
    raise exc


def _patch_value(
    kube: Kubectl,
    kind: str,
    name: str,
    key: str,
    value: str,
) -> None:
    stored = base64.b64encode(value.encode()).decode() if kind == "secret" else value
    try:
        kube.run(
            [
                "patch", kind, name, "-n", kube.namespace, "--type=merge", "-p",
                json.dumps({"data": {key: stored}}),
            ]
        )
    except KubectlError as exc:
        _candidate_mutation_failure(kube, exc, f"{kind}/{name} update")


def _current_value(kube: Kubectl, kind: str, name: str, key: str) -> str:
    raw = kube.get(kind, name).get("data", {}).get(key)
    if kind != "secret":
        return str(raw)
    try:
        return base64.b64decode(raw, validate=True).decode("utf-8")
    except Exception as exc:
        raise RequirementFailure(f"{name}/{key} is not valid base64 UTF-8") from exc


def _live_update(
    kube: Kubectl,
    kind: str,
    name: str,
    key: str,
    changed: str,
    original: str,
    assertions: Callable[[str], bool],
    *,
    timeout: int,
    controllers: list[tuple[str, str]],
    selectors: list[str],
    changed_check: Callable[[], None] | None = None,
    restored_check: Callable[[], None] | None = None,
) -> str:
    generations = _generations(kube, controllers)
    uids = {selector: _uids(kube, selector) for selector in selectors}
    try:
        _patch_value(kube, kind, name, key, changed)
        kube.wait_until(lambda: assertions(changed), f"{name} live update", timeout, 2)
        if changed_check is not None:
            changed_check()
        require(_generations(kube, controllers) == generations, "controller generation changed during live update")
        require({selector: _uids(kube, selector) for selector in selectors} == uids, "pod UID changed during live update")
    finally:
        if _current_value(kube, kind, name, key) != original:
            try:
                _patch_value(kube, kind, name, key, original)
            except Exception as exc:
                raise EvaluationInfrastructureError(f"could not restore {kind}/{name}") from exc
        kube.wait_until(lambda: assertions(original), f"{name} restoration", timeout, 2)
        if restored_check is not None:
            restored_check()
        require(_generations(kube, controllers) == generations, "controller generation changed during restoration")
        require({selector: _uids(kube, selector) for selector in selectors} == uids, "pod UID changed during restoration")
    return f"{name} changed and was restored without rollout"


def _bounded_fetch(text: str, seconds: int = 5) -> None:
    normalized = " ".join(text.lower().split())
    patterns = (
        rf"wget[^;\n]*(?:--timeout(?:=|\s+)|-t\s*){seconds}(?:\s|$)",
        rf"\btimeout\s+{seconds}(?:s)?\s+(?:wget|curl)\b",
        rf"curl[^;\n]*--connect-timeout(?:=|\s+){seconds}(?:\s|$)",
        rf"curl[^;\n]*--max-time(?:=|\s+){seconds}(?:\s|$)",
    )
    require(any(re.search(pattern, normalized) for pattern in patterns), f"a {seconds}-second-bounded request is absent")


def _bounded_attempts(text: str, maximum: int, sleep: int = 2) -> None:
    normalized = " ".join(text.split())
    require(re.search(r"\b(?:exit\s+[1-9][0-9]*|false)\b", normalized) is not None, "retry exhaustion does not fail non-zero")
    require(re.search(rf"\bsleep\s+{sleep}(?:\s|;|$)", normalized) is not None, f"retry interval must be {sleep} seconds")
    counts: list[int] = []
    for match in re.finditer(r"\bseq\s+(?:(0|1)\s+)?([0-9]+)\b", normalized):
        start, end = match.groups()
        counts.append(int(end) + 1 if start == "0" else int(end))
    for match in re.finditer(r"-(?:ge|lt)\s+([0-9]+)\b", normalized):
        counts.append(int(match.group(1)))
    for match in re.finditer(r"-le\s+([0-9]+)\b", normalized):
        counts.append(int(match.group(1)))
    for match in re.finditer(r"-gt\s+([0-9]+)\b", normalized):
        counts.append(int(match.group(1)) + 1)
    require(any(1 <= count <= maximum for count in counts), f"no attempt bound at or below {maximum} was found")


def _job_complete(kube: Kubectl, name: str, timeout: int, expected_log: str) -> str:
    kube.wait_until(
        lambda: job_terminal_condition(kube.get("job", name)) is not None,
        f"Job {name} terminal condition",
        timeout,
        1,
    )
    require(job_terminal_condition(kube.get("job", name)) == "Complete", f"{name} did not Complete")
    pods = kube.list("pods", f"job-name={name}")
    require(len(pods) == 1, f"{name} must have exactly one pod")
    logs = kube.run(["logs", pods[0]["metadata"]["name"], "-n", kube.namespace]).stdout
    require(logs == expected_log, f"{name} logs differ")
    return f"{name} completed with exact logs"


def _job_contract(kube: Kubectl, name: str, deadline: int, expected_log: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    job = kube.get("job", name)
    spec = job.get("spec", {})
    require(spec.get("backoffLimit") == 0, f"{name} backoffLimit differs")
    require(spec.get("activeDeadlineSeconds") == deadline, f"{name} activeDeadlineSeconds differs")
    require(spec.get("completions", 1) == 1 and spec.get("parallelism", 1) == 1, f"{name} completion shape differs")
    workload = _job_workload(job)
    require(pod_spec(workload).get("restartPolicy") == "Never", f"{name} restartPolicy differs")
    container = _sole_container(workload)
    require(container.get("image") == "busybox:1.36.1", f"{name} image differs")
    return workload, container, _job_complete(kube, name, deadline, expected_log)


def _cron_contract(kube: Kubectl, name: str, schedule: str, deadline: int) -> tuple[dict[str, Any], dict[str, Any]]:
    cron = kube.get("cronjob", name)
    spec = cron.get("spec", {})
    require(spec.get("schedule") == schedule, f"{name} schedule differs")
    require(spec.get("suspend") is True, f"{name} must be suspended")
    require(spec.get("concurrencyPolicy") == "Forbid", f"{name} concurrencyPolicy differs")
    require(spec.get("successfulJobsHistoryLimit") == 1, f"{name} successful history differs")
    require(spec.get("failedJobsHistoryLimit") == 1, f"{name} failed history differs")
    job_spec = spec.get("jobTemplate", {}).get("spec", {})
    require(job_spec.get("activeDeadlineSeconds") == deadline, f"{name} deadline differs")
    workload = _cron_workload(cron)
    require(pod_spec(workload).get("restartPolicy") == "Never", f"{name} restartPolicy differs")
    container = _sole_container(workload)
    require(container.get("image") == "busybox:1.36.1", f"{name} image differs")
    return workload, container


def _run_cron(kube: Kubectl, cron: str, suffix: str, timeout: int, expected_log: str) -> str:
    name = f"aipc-eval-{suffix}"
    try:
        kube.delete("job", name)
        try:
            kube.run(["create", "job", name, f"--from=cronjob/{cron}", "-n", kube.namespace])
        except KubectlError as exc:
            kube._raise_evaluator_creation_failure(exc, f"probe Job {name}")
        return _job_complete(kube, name, timeout, expected_log)
    finally:
        kube.delete("job", name)


def _snapshot(kube: Kubectl, kind: str, name: str) -> dict[str, Any]:
    current = kube.get(kind, name)
    result = {"apiVersion": current["apiVersion"], "kind": current["kind"]}
    result["metadata"] = {"name": name, "namespace": kube.namespace}
    for key in ("spec", "roleRef", "subjects", "data", "binaryData", "type", "immutable"):
        if key in current:
            result[key] = current[key]
    return result


def _restore_object(kube: Kubectl, item: dict[str, Any], description: str) -> None:
    try:
        kube.run(["apply", "-f", "-"], input_text=json.dumps(item))
    except KubectlError as exc:
        raise EvaluationInfrastructureError(f"could not restore {description}") from exc


def _replacement_continuity(
    kube: Kubectl,
    selector: str,
    service: str,
    expected: str,
    labels: dict[str, str],
    timeout: int,
) -> str:
    pods = _ready_pods(kube, selector)
    require(len(pods) == 2, "continuity check requires two Ready replicas")
    victim = pods[0]
    old_uid = victim["metadata"]["uid"]
    kube.run(["delete", "pod", victim["metadata"]["name"], "-n", kube.namespace, "--wait=true"])
    kube.wait_until(lambda: old_uid not in _endpoint_uids(kube, service), "deleted endpoint removal", 30, 1)
    encoded = base64.b64encode(expected.encode()).decode()
    command = (
        'i=0; while [ "$i" -lt 10 ]; do '
        f'test "$(wget -q -T 5 -O - http://{service} | base64 | tr -d \'\\n\')" = {encoded}; '
        'i=$((i+1)); sleep 1; done'
    )
    kube.run_probe_pod(f"aipc-eval-{service}-continuity"[:63], command, labels=labels, timeout_seconds=35)
    kube.wait_until(
        lambda: any(
            _ready(pod) and pod.get("metadata", {}).get("uid") not in {"", old_uid}
            for pod in kube.list("pods", selector)
        ),
        "different-UID Ready replacement",
        timeout,
        1,
    )
    return "all ten exact requests succeeded during a different-UID replacement"


def _exact_role(kube: Kubectl, name: str) -> str:
    role = kube.get("role", name)
    actual: set[tuple[str, str, str, str]] = set()
    for rule in role.get("rules", []) or []:
        require(not rule.get("nonResourceURLs"), f"{name} has non-resource access")
        for group in rule.get("apiGroups", []) or []:
            for resource in rule.get("resources", []) or []:
                for resource_name in rule.get("resourceNames", []) or []:
                    for verb in rule.get("verbs", []) or []:
                        actual.add((group, resource, resource_name, verb))
    require(actual == {("", "secrets", "auth-content", "get")}, f"{name} permissions differ")
    return f"{name} grants only get on secrets/auth-content"


def _binding(kube: Kubectl, name: str, role: str, service_account: str) -> str:
    item = kube.get("rolebinding", name)
    reference = item.get("roleRef", {})
    require(
        reference.get("apiGroup") == "rbac.authorization.k8s.io"
        and reference.get("kind") == "Role"
        and reference.get("name") == role,
        f"{name} roleRef differs",
    )
    subjects = item.get("subjects", []) or []
    require(len(subjects) == 1, f"{name} must bind exactly one subject")
    subject = subjects[0]
    require(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace", kube.namespace) == kube.namespace,
        f"{name} subject differs",
    )
    return f"{name} binds only {service_account} to {role}"


def _command_has(container: dict[str, Any], *tokens: str) -> str:
    text = command_text(container)
    for token in tokens:
        require(token in text, f"container program does not reference {token!r}")
    return text


def _medium6_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def deployment() -> str:
        item = kube.get("deployment", "signal-web")
        _labels_include(item, {"app": "signal-web"})
        _rolling(item, replicas=2, min_ready=5, deadline=120)
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "signal-web image differs")
        _mount_source(item, container, "/page", "configMap", "signal-page")
        _mount_source(item, container, "/code", "secret", "signal-code")
        _emptydir(item, container, "/www")
        _port(container, "signal-http", 8080)
        _http_readiness(container, "signal-http", 8080, period=5, failure=2)
        _command_has(container, "/page", "/code", "/www", "sleep 2", "rm")
        require(_ready_count(kube, "app=signal-web") == 2, "signal-web does not have two Ready replicas")
        return "signal-web has the declared live projections, serving loop, rollout, and readiness"

    def service() -> str:
        _service(kube, "signal-svc", {"app": "signal-web"}, 80, "signal-http", 8080)
        return _body_probe(
            kube,
            "aipc-eval-signal-body",
            "http://signal-svc",
            "signal=v1;code=green\n",
            labels={"access": "signal"},
        )

    def ingress() -> str:
        return _live_ingress(
            kube,
            [("aipc-eval-signal-allow", "http://signal-svc", {"access": "signal"})],
            [("aipc-eval-signal-deny", "http://signal-svc")],
        )

    def governance() -> str:
        return _pdb(kube, "signal-pdb", {"app": "signal-web"}, 1) + "; " + _limit_range(kube, "signal-defaults")

    def update() -> str:
        container = _sole_container(kube.get("deployment", "signal-web"))["name"]

        def bodies(value: str) -> bool:
            return _all_pod_bodies(
                kube,
                "app=signal-web",
                container,
                8080,
                f"signal=v1;code={value}\n",
                2,
            )

        def changed_service() -> None:
            _body_probe(
                kube,
                "aipc-eval-signal-amber",
                "http://signal-svc",
                "signal=v1;code=amber\n",
                labels={"access": "signal"},
            )

        def restored_service() -> None:
            _body_probe(
                kube,
                "aipc-eval-signal-green",
                "http://signal-svc",
                "signal=v1;code=green\n",
                labels={"access": "signal"},
            )

        return _live_update(
            kube,
            "secret",
            "signal-code",
            "CODE",
            "amber",
            "green",
            bodies,
            timeout=120,
            controllers=[("deployment", "signal-web")],
            selectors=["app=signal-web"],
            changed_check=changed_service,
            restored_check=restored_service,
        )

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-signal-ns", candidate_path)),
        ("R02", "immutable signal page", lambda: _exact_configmap(kube, "signal-page", {"index.html": "signal=v1\n"}, immutable=True)),
        ("R03", "mutable signal code", lambda: _exact_secret(kube, "signal-code", "CODE", "green", immutable=False)),
        ("R04", "signal Deployment behavior", deployment),
        ("R05", "signal Service exact response", service),
        ("R06", "signal pod hardening", lambda: _hardened(kube.get("deployment", "signal-web"), "signal-web", {"/www"})),
        ("R07", "signal live ingress isolation", ingress),
        ("R08", "signal disruption and resource defaults", governance),
        ("R09", "live Secret projection and restoration", update),
        ("R10", "replacement continuity", lambda: _replacement_continuity(kube, "app=signal-web", "signal-svc", "signal=v1;code=green\n", {"access": "signal"}, 60)),
    ]


def _medium7_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def daemonset() -> str:
        item = kube.get("daemonset", "node-banner")
        _labels_include(item, {"app": "node-banner"})
        spec = pod_spec(item)
        require(not spec.get("nodeSelector"), "node-banner must not pin itself with nodeSelector")
        require(not spec.get("tolerations"), "node-banner must not add template tolerations")
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "node-banner image differs")
        _secret_env(container, "CHECK", "node-banner-key", "CHECK")
        _mount_source(item, container, "/srv", "configMap", "node-banner-page")
        _port(container, "node-http", 8090)
        _http_readiness(container, "node-http", 8090, period=5, failure=3)
        _command_has(container, "CHECK", "node-ok")
        require(_ready_count(kube, "app=node-banner") == 1, "reference cluster must have one Ready banner pod")
        return "ordinary DaemonSet has the declared projected page, startup gate, listener, and readiness"

    def service() -> str:
        _service(kube, "node-banner-svc", {"app": "node-banner"}, 80, "node-http", 8090)
        return _body_probe(kube, "aipc-eval-node-body", "http://node-banner-svc", "node-banner=v1\n", labels={"access": "node-banner"})

    def update() -> str:
        container = _sole_container(kube.get("daemonset", "node-banner"))["name"]
        bodies = lambda value: _all_pod_bodies(kube, "app=node-banner", container, 8090, value, 1)
        return _live_update(
            kube,
            "configmap",
            "node-banner-page",
            "index.html",
            "node-banner=v2\n",
            "node-banner=v1\n",
            bodies,
            timeout=120,
            controllers=[("daemonset", "node-banner")],
            selectors=["app=node-banner"],
            changed_check=lambda: _body_probe(kube, "aipc-eval-node-v2", "http://node-banner-svc", "node-banner=v2\n", labels={"access": "node-banner"}) and None,
            restored_check=lambda: _body_probe(kube, "aipc-eval-node-v1", "http://node-banner-svc", "node-banner=v1\n", labels={"access": "node-banner"}) and None,
        )

    def replacement() -> str:
        pods = _ready_pods(kube, "app=node-banner")
        require(len(pods) == 1, "replacement check requires one Ready banner pod")
        old = pods[0]
        old_uid = old["metadata"]["uid"]
        kube.run(["delete", "pod", old["metadata"]["name"], "-n", kube.namespace, "--wait=true"])
        kube.wait_until(
            lambda: any(_ready(pod) and pod.get("metadata", {}).get("uid") not in {"", old_uid} for pod in kube.list("pods", "app=node-banner")),
            "different-UID Ready banner replacement",
            90,
            1,
        )
        _body_probe(kube, "aipc-eval-node-recovery", "http://node-banner-svc", "node-banner=v1\n", labels={"access": "node-banner"})
        return "a different-UID Ready DaemonSet pod restored the exact Service response"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-node-ns", candidate_path)),
        ("R02", "mutable banner page", lambda: _exact_configmap(kube, "node-banner-page", {"index.html": "node-banner=v1\n"}, immutable=False)),
        ("R03", "immutable banner key", lambda: _exact_secret(kube, "node-banner-key", "CHECK", "node-ok", immutable=True)),
        ("R04", "node banner DaemonSet", daemonset),
        ("R05", "node banner Service", service),
        ("R06", "node banner hardening", lambda: _hardened(kube.get("daemonset", "node-banner"), "node-banner", set())),
        ("R07", "node banner ingress isolation", lambda: _live_ingress(kube, [("aipc-eval-node-allow", "http://node-banner-svc", {"access": "node-banner"})], [("aipc-eval-node-deny", "http://node-banner-svc")])),
        ("R08", "node banner availability budget", lambda: _pdb(kube, "node-banner-pdb", {"app": "node-banner"}, 1)),
        ("R09", "live banner projection and restoration", update),
        ("R10", "DaemonSet pod replacement", replacement),
    ]


def _product_program(container: dict[str, Any], output: str) -> str:
    text = _command_has(container, "VALUE", "MULTIPLIER", "MODE", "scale", output)
    require(
        "*" in text or re.search(r"\b(?:expr|awk)\b", text) is not None,
        "batch program does not calculate the product",
    )
    require(
        re.search(r"\b(?:exit\s+[1-9][0-9]*|false|test)\b|\[", text) is not None,
        "batch program does not fail when its mode, inputs, or result are wrong",
    )
    return text


def _medium8_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def build() -> str:
        workload, container, result = _job_contract(kube, "report-build", 60, "report=36\n")
        _config_env(container, "VALUE", "report-input", "VALUE")
        _config_env(container, "MULTIPLIER", "report-input", "MULTIPLIER")
        _secret_env(container, "MODE", "report-mode", "MODE")
        _product_program(container, "report=36")
        return result

    def cron() -> str:
        _, container = _cron_contract(kube, "report-audit", "11 */3 * * *", 60)
        workload = _cron_workload(kube.get("cronjob", "report-audit"))
        _projected_input(workload, container, "VALUE", "configMap", "report-input", "VALUE")
        _projected_input(
            workload,
            container,
            "MULTIPLIER",
            "configMap",
            "report-input",
            "MULTIPLIER",
        )
        _projected_input(workload, container, "MODE", "secret", "report-mode", "MODE")
        _product_program(container, "report-audit=36")
        return "report-audit has the declared schedule, inputs, calculation, and execution bounds"

    def hardening() -> str:
        job = _hardened(_job_workload(kube.get("job", "report-build")), "report-build", set())
        cron_result = _hardened(_cron_workload(kube.get("cronjob", "report-audit")), "report-audit", set())
        return job + "; " + cron_result

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-report-ns", candidate_path)),
        ("R02", "immutable calculation inputs", lambda: _exact_configmap(kube, "report-input", {"VALUE": "9", "MULTIPLIER": "4"}, immutable=True)),
        ("R03", "immutable calculation mode", lambda: _exact_secret(kube, "report-mode", "MODE", "scale", immutable=True)),
        ("R04", "one-shot report calculation", build),
        ("R05", "report quota", lambda: _quota(kube, "report-quota", {"pods": "6", "count/jobs.batch": "4", "count/cronjobs.batch": "2", "requests.cpu": "1", "requests.memory": "512Mi", "limits.cpu": "2", "limits.memory": "1Gi"})),
        ("R06", "report container defaults", lambda: _limit_range(kube, "report-defaults")),
        ("R07", "suspended audit schedule", cron),
        ("R08", "executable audit template", lambda: _run_cron(kube, "report-audit", "report-audit", 60, "report-audit=36\n")),
        ("R09", "batch pod hardening", hardening),
    ]


def _medium9_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def deployment() -> str:
        item = kube.get("deployment", "release-web")
        _labels_include(item, {"app": "release-web"})
        _rolling(item, replicas=2, min_ready=5, deadline=120)
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "release-web image differs")
        _mount_source(item, container, "/page", "configMap", "release-page")
        _mount_source(item, container, "/channel", "secret", "release-channel")
        _emptydir(item, container, "/www")
        _port(container, "release-http", 8080)
        _http_readiness(container, "release-http", 8080, period=5, failure=2)
        _command_has(container, "/page", "/channel", "/www", "sleep 2")
        require(_ready_count(kube, "app=release-web") == 2, "release-web does not have two Ready replicas")
        return "release-web has the declared projections, serving loop, rollout, and readiness"

    def service() -> str:
        _service(kube, "release-svc", {"app": "release-web"}, 80, "release-http", 8080)
        return _body_probe(kube, "aipc-eval-release-body", "http://release-svc", "release=stable;channel=blue\n", labels={"access": "release"})

    def cron() -> str:
        workload, container = _cron_contract(kube, "release-audit", "*/9 * * * *", 60)
        require(pod_labels(workload).get("access") == "release", "release-audit pod label access=release is missing")
        text = _command_has(container, "release-svc", "release=stable;channel=blue", "release-audit-ok")
        _bounded_attempts(text, 20, 2)
        return "release-audit has the declared label, schedule, bounded retry, and exact assertion"

    def hardening() -> str:
        return _hardened(kube.get("deployment", "release-web"), "release-web", {"/www"}) + "; " + _hardened(_cron_workload(kube.get("cronjob", "release-audit")), "release-audit", set())

    def isolation_and_pdb() -> str:
        live = _live_ingress(kube, [("aipc-eval-release-allow", "http://release-svc", {"access": "release"})], [("aipc-eval-release-deny", "http://release-svc")])
        return live + "; " + _pdb(kube, "release-pdb", {"app": "release-web"}, 1)

    def update() -> str:
        container = _sole_container(kube.get("deployment", "release-web"))["name"]
        bodies = lambda value: _all_pod_bodies(kube, "app=release-web", container, 8080, f"release=stable;channel={value}\n", 2)
        return _live_update(
            kube,
            "secret",
            "release-channel",
            "CHANNEL",
            "green",
            "blue",
            bodies,
            timeout=120,
            controllers=[("deployment", "release-web")],
            selectors=["app=release-web"],
            changed_check=lambda: _body_probe(kube, "aipc-eval-release-green", "http://release-svc", "release=stable;channel=green\n", labels={"access": "release"}) and None,
            restored_check=lambda: _body_probe(kube, "aipc-eval-release-blue", "http://release-svc", "release=stable;channel=blue\n", labels={"access": "release"}) and None,
        )

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-release-ns", candidate_path)),
        ("R02", "immutable release page", lambda: _exact_configmap(kube, "release-page", {"index.html": "release=stable\n"}, immutable=True)),
        ("R03", "mutable release channel", lambda: _exact_secret(kube, "release-channel", "CHANNEL", "blue", immutable=False)),
        ("R04", "release Deployment behavior", deployment),
        ("R05", "release Service exact response", service),
        ("R06", "suspended release audit", cron),
        ("R07", "release and audit hardening", hardening),
        ("R08", "release isolation and PDB", isolation_and_pdb),
        ("R09", "live channel projection and restoration", update),
        ("R10", "executable release audit", lambda: _run_cron(kube, "release-audit", "release-audit", 60, "release-audit-ok\n")),
    ]


def _medium10_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def statefulset() -> str:
        item = kube.get("statefulset", "cache")
        _labels_include(item, {"app": "cache"})
        spec = item.get("spec", {})
        require(spec.get("replicas") == 2 and spec.get("serviceName") == "cache-headless", "cache identity differs")
        require(spec.get("podManagementPolicy") == "OrderedReady", "cache must use OrderedReady")
        require(spec.get("persistentVolumeClaimRetentionPolicy") == {"whenDeleted": "Retain", "whenScaled": "Retain"}, "cache PVC retention differs")
        update = spec.get("updateStrategy", {})
        require(update.get("type", "RollingUpdate") == "RollingUpdate", "cache update strategy differs")
        require(update.get("rollingUpdate", {}).get("partition", 0) == 0, "cache partition differs")
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "cache image differs")
        _mount_source(item, container, "/config", "configMap", "cache-prefix")
        _mount_source(item, container, "/mode", "secret", "cache-mode")
        data_mount = find_mount(container, "/data")
        require(data_mount is not None, "cache /data mount is absent")
        assert_claim(volume_claim_template(item, data_mount.get("name", "")), "64Mi")
        _port(container, "cache-http", 8080)
        _http_readiness(container, "cache-http", 8080, period=5, failure=None)
        text = _command_has(container, "/config", "/mode", "/data/index.html", "sleep 3")
        require(
            "HOSTNAME" in text or re.search(r"\bhostname\b", text) is not None,
            "cache program does not derive the pod hostname",
        )
        require(_ready_count(kube, "app=cache") == 2, "cache does not have two Ready members")
        require(
            _all_pod_bodies(
                kube,
                "app=cache",
                container.get("name", ""),
                8080,
                lambda pod: f"cache-{pod}-ready\n",
                2,
            ),
            "cache members do not serve their exact initial ordinal bodies",
        )
        return "cache StatefulSet has the declared identity, claims, projections, serving loop, and readiness"

    def update() -> str:
        container = _sole_container(kube.get("statefulset", "cache"))["name"]

        def bodies(value: str) -> bool:
            return _all_pod_bodies(
                kube,
                "app=cache",
                container,
                8080,
                lambda pod: f"cache-{pod}-{value}\n",
                2,
            )

        def service_value(value: str) -> None:
            for ordinal in (0, 1):
                _body_probe(
                    kube,
                    f"aipc-eval-cache-{ordinal}-{value}"[:63],
                    f"http://cache-{ordinal}.cache-headless:8080",
                    f"cache-cache-{ordinal}-{value}\n",
                    labels={"access": "cache"},
                )

        return _live_update(
            kube,
            "secret",
            "cache-mode",
            "SUFFIX",
            "warm",
            "ready",
            bodies,
            timeout=120,
            controllers=[("statefulset", "cache")],
            selectors=["app=cache"],
            changed_check=lambda: service_value("warm"),
            restored_check=lambda: service_value("ready"),
        )

    def retained() -> str:
        pod = kube.get("pod", "cache-0")
        require(_ready(pod), "cache-0 is not Ready")
        old_uid = pod.get("metadata", {}).get("uid", "")
        container = _sole_container(kube.get("statefulset", "cache"))["name"]
        kube.exec("cache-0", "printf 'retained\\n' > /data/oracle-marker", container)
        try:
            kube.run(["delete", "pod", "cache-0", "-n", kube.namespace, "--wait=true"])

            def recovered() -> bool:
                current = kube.get("pod", "cache-0")
                return _ready(current) and current.get("metadata", {}).get("uid") not in {"", old_uid}

            kube.wait_until(recovered, "different-UID Ready cache-0", 90, 1)
            kube.exec("cache-0", 'test "$(cat /data/oracle-marker)" = retained', container)
            kube.wait_until(lambda: _endpoint_count(kube, "cache-headless") == 2, "two Ready cache endpoints", 30, 1)
        finally:
            kube.run(["exec", "-n", kube.namespace, "cache-0", "-c", container, "--", "rm", "-f", "/data/oracle-marker"], check=False, timeout=15)
        return "cache-0 retained the exact evaluator marker across a different-UID replacement"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-cache-ns", candidate_path)),
        ("R02", "immutable cache prefix", lambda: _exact_configmap(kube, "cache-prefix", {"PREFIX": "cache"}, immutable=True)),
        ("R03", "mutable cache mode", lambda: _exact_secret(kube, "cache-mode", "SUFFIX", "ready", immutable=False)),
        ("R04", "cache headless Service", lambda: _service(kube, "cache-headless", {"app": "cache"}, 8080, "cache-http", 8080, headless=True) and "cache-headless has the declared identity and port"),
        ("R05", "cache StatefulSet and storage", statefulset),
        ("R06", "cache hardening", lambda: _hardened(kube.get("statefulset", "cache"), "cache", {"/data"})),
        ("R07", "cache stable-name isolation", lambda: _live_ingress(kube, [("aipc-eval-cache-zero", "http://cache-0.cache-headless:8080", {"access": "cache"}), ("aipc-eval-cache-one", "http://cache-1.cache-headless:8080", {"access": "cache"})], [("aipc-eval-cache-deny", "http://cache-0.cache-headless:8080")])),
        ("R08", "cache PDB and quota", lambda: _pdb(kube, "cache-pdb", {"app": "cache"}, 1) + "; " + _quota(kube, "cache-quota", {"pods": "6", "persistentvolumeclaims": "3", "requests.storage": "256Mi", "services": "2"})),
        ("R09", "live cache mode projection", update),
        ("R10", "retained ordinal replacement", retained),
    ]


def _medium11_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def deployment() -> str:
        item = kube.get("deployment", "notice-web")
        _labels_include(item, {"app": "notice-web"})
        _rolling(item, replicas=2, min_ready=5, deadline=120)
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "notice-web image differs")
        _mount_source(item, container, "/base", "configMap", "notice-base")
        _secret_env(container, "TOKEN", "notice-token", "TOKEN")
        _emptydir(item, container, "/www")
        _port(container, "notice-http", 8080)
        _http_readiness(container, "notice-http", 8080, period=5, failure=2)
        _command_has(container, "/base", "BASE", "TOKEN", "/www/index.html", "sleep 2")
        require(_ready_count(kube, "app=notice-web") == 2, "notice-web does not have two Ready replicas")
        return "notice-web has the declared projected base, token, rendering loop, rollout, and readiness"

    def service() -> str:
        _service(kube, "notice-svc", {"app": "notice-web"}, 80, "notice-http", 8080)
        return _body_probe(kube, "aipc-eval-notice-body", "http://notice-svc", "notice-valid\n", labels={"access": "notice"})

    def verifier() -> str:
        workload, container, result = _job_contract(kube, "notice-verify", 60, "notice-verify-ok\n")
        require(pod_labels(workload).get("access") == "notice", "notice-verify pod label access=notice is missing")
        text = _command_has(container, "notice-svc", "notice-valid", "notice-verify-ok")
        _bounded_attempts(text, 20, 2)
        return result

    def hardening() -> str:
        return _hardened(kube.get("deployment", "notice-web"), "notice-web", {"/www"}) + "; " + _hardened(_job_workload(kube.get("job", "notice-verify")), "notice-verify", set())

    def update() -> str:
        container = _sole_container(kube.get("deployment", "notice-web"))["name"]
        bodies = lambda value: _all_pod_bodies(kube, "app=notice-web", container, 8080, f"{value}-valid\n", 2)
        return _live_update(
            kube,
            "configmap",
            "notice-base",
            "BASE",
            "alert",
            "notice",
            bodies,
            timeout=120,
            controllers=[("deployment", "notice-web")],
            selectors=["app=notice-web"],
            changed_check=lambda: _body_probe(kube, "aipc-eval-notice-alert", "http://notice-svc", "alert-valid\n", labels={"access": "notice"}) and None,
            restored_check=lambda: _body_probe(kube, "aipc-eval-notice-normal", "http://notice-svc", "notice-valid\n", labels={"access": "notice"}) and None,
        )

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-notice-ns", candidate_path)),
        ("R02", "mutable notice base", lambda: _exact_configmap(kube, "notice-base", {"BASE": "notice"}, immutable=False)),
        ("R03", "immutable notice token", lambda: _exact_secret(kube, "notice-token", "TOKEN", "valid", immutable=True)),
        ("R04", "notice Deployment behavior", deployment),
        ("R05", "notice Service exact response", service),
        ("R06", "bounded one-shot verifier", verifier),
        ("R07", "notice and verifier hardening", hardening),
        ("R08", "notice live ingress isolation", lambda: _live_ingress(kube, [("aipc-eval-notice-allow", "http://notice-svc", {"access": "notice"})], [("aipc-eval-notice-deny", "http://notice-svc")])),
        ("R09", "notice quota", lambda: _quota(kube, "notice-quota", {"pods": "8", "services": "2", "count/jobs.batch": "3", "requests.cpu": "1", "requests.memory": "512Mi", "limits.cpu": "2", "limits.memory": "1Gi"})),
        ("R10", "live notice projection and restoration", update),
    ]


def _medium12_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def deployment() -> str:
        item = kube.get("deployment", "status-web")
        _labels_include(item, {"app": "status-web"})
        _rolling(item, replicas=2, min_ready=5, deadline=120, history=2)
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "status-web image differs")
        _mount_source(item, container, "/mode", "configMap", "status-mode")
        _emptydir(item, container, "/www")
        _port(container, "status-http", 8080)
        _http_readiness(container, "status-http", 8080, period=5, failure=2)
        _command_has(container, "/mode", "MODE", "ready", "/www/index.html", "rm", "sleep 2")
        require(_ready_count(kube, "app=status-web") == 2, "status-web does not have two Ready replicas")
        return "status-web has the declared mode gate, output withdrawal, rollout, and readiness"

    def service() -> str:
        _service(kube, "status-svc", {"app": "status-web"}, 80, "status-http", 8080)
        return _body_probe(kube, "aipc-eval-status-body", "http://status-svc", "status=ready\n", labels={"access": "status"})

    def transition() -> str:
        container = _sole_container(kube.get("deployment", "status-web"))["name"]

        def state(value: str) -> bool:
            if value == "hold":
                return _ready_count(kube, "app=status-web") == 0 and _endpoint_count(kube, "status-svc") == 0
            return (
                _endpoint_count(kube, "status-svc") == 2
                and _all_pod_bodies(kube, "app=status-web", container, 8080, "status=ready\n", 2)
            )

        return _live_update(
            kube,
            "configmap",
            "status-mode",
            "MODE",
            "hold",
            "ready",
            state,
            timeout=120,
            controllers=[("deployment", "status-web")],
            selectors=["app=status-web"],
            restored_check=lambda: _body_probe(kube, "aipc-eval-status-restored", "http://status-svc", "status=ready\n", labels={"access": "status"}) and None,
        )

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-status-ns", candidate_path)),
        ("R02", "mutable status mode", lambda: _exact_configmap(kube, "status-mode", {"MODE": "ready"}, immutable=False)),
        ("R03", "readiness-sensitive status Deployment", deployment),
        ("R04", "status Service exact response", service),
        ("R05", "status pod hardening", lambda: _hardened(kube.get("deployment", "status-web"), "status-web", {"/www"})),
        ("R06", "status live ingress isolation", lambda: _live_ingress(kube, [("aipc-eval-status-allow", "http://status-svc", {"access": "status"})], [("aipc-eval-status-deny", "http://status-svc")])),
        ("R07", "status availability budget", lambda: _pdb(kube, "status-pdb", {"app": "status-web"}, 1)),
        ("R08", "status container defaults", lambda: _limit_range(kube, "status-defaults")),
        ("R09", "reversible readiness withdrawal", transition),
        ("R10", "status replacement continuity", lambda: _replacement_continuity(kube, "app=status-web", "status-svc", "status=ready\n", {"access": "status"}, 60)),
    ]


def _medium13_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def service_account() -> str:
        account = kube.get("serviceaccount", "auth-reader")
        require(account.get("automountServiceAccountToken") is True, "auth-reader token automount must be true")
        return "auth-reader explicitly enables its projected API token"

    def role() -> str:
        result = _exact_role(kube, "auth-content-reader")
        require(kube.auth_can_i("auth-reader", "get", "secret/auth-content"), "auth-reader cannot get auth-content")
        require(not kube.auth_can_i("auth-reader", "list", "secrets"), "auth-reader can list Secrets")
        require(not kube.auth_can_i("auth-reader", "get", "secrets"), "auth-reader can get arbitrary Secrets")
        require(not kube.auth_can_i("auth-reader", "get", "configmaps"), "auth-reader can get ConfigMaps")
        return result + "; effective access has no tested widening"

    def deployment() -> str:
        item = kube.get("deployment", "auth-web")
        _labels_include(item, {"app": "auth-web"})
        _rolling(item, replicas=2, min_ready=5, deadline=150)
        require(pod_spec(item).get("serviceAccountName") == "auth-reader", "auth-web does not use auth-reader")
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "auth-web image differs")
        _mount_source(item, container, "/content", "secret", "auth-content", key_path=("DATA", "index.html"))
        _emptydir(item, container, "/work")
        _port(container, "auth-http", 8080)
        text = _command_has(
            container,
            "https://",
            "secrets/auth-content",
            "/var/run/secrets/kubernetes.io/serviceaccount/token",
            "/work/authorized",
            "auth-ready",
            "rm",
            "sleep 3",
        )
        _bounded_fetch(text, 5)
        probe = container.get("readinessProbe", {})
        require(probe.get("periodSeconds", 10) == 5 and probe.get("failureThreshold", 3) == 2, "auth readiness timing differs")
        command = " ".join(probe.get("exec", {}).get("command", []) or [])
        require("/work/authorized" in command, "auth readiness does not require the authorization marker")
        require("127.0.0.1" in command and "auth-ready" in command, "auth readiness does not require the exact local body")
        require(_ready_count(kube, "app=auth-web") == 2, "auth-web does not have two Ready replicas")
        return "auth-web has the narrow API check, marker gate, exact local readiness, and two Ready replicas"

    def service() -> str:
        _service(kube, "auth-svc", {"app": "auth-web"}, 80, "auth-http", 8080)
        return _body_probe(kube, "aipc-eval-auth-body", "http://auth-svc", "auth-ready", labels={})

    def authorization_outage() -> str:
        saved = _snapshot(kube, "rolebinding", "auth-content-reader-binding")
        try:
            kube.run(["delete", "rolebinding", "auth-content-reader-binding", "-n", kube.namespace, "--wait=true"])
            kube.wait_until(
                lambda: _ready_count(kube, "app=auth-web") == 0 and _endpoint_count(kube, "auth-svc") == 0,
                "authorization-sensitive readiness loss",
                60,
                2,
            )
        finally:
            _restore_object(kube, saved, "auth-content-reader-binding")
            kube.wait_until(
                lambda: _ready_count(kube, "app=auth-web") == 2 and _endpoint_count(kube, "auth-svc") == 2,
                "authorization recovery",
                120,
                2,
            )
        _body_probe(kube, "aipc-eval-auth-restored", "http://auth-svc", "auth-ready", labels={})
        return "RoleBinding loss withdrew all endpoints and exact restoration recovered them"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-auth-ns", candidate_path)),
        ("R02", "immutable authenticated content", lambda: _exact_secret(kube, "auth-content", "DATA", "auth-ready", immutable=True)),
        ("R03", "authenticated service account", service_account),
        ("R04", "exact named-Secret permission", role),
        ("R05", "exact service-account binding", lambda: _binding(kube, "auth-content-reader-binding", "auth-content-reader", "auth-reader")),
        ("R06", "authorization-sensitive Deployment", deployment),
        ("R07", "authenticated exact Service body", service),
        ("R08", "authenticated endpoint PDB", lambda: _pdb(kube, "auth-pdb", {"app": "auth-web"}, 1)),
        ("R09", "authenticated pod hardening", lambda: _hardened(kube.get("deployment", "auth-web"), "auth-web", {"/work"})),
        ("R10", "reversible authorization outage", authorization_outage),
        ("R11", "authorized replacement continuity", lambda: _replacement_continuity(kube, "app=auth-web", "auth-svc", "auth-ready", {}, 60)),
    ]


def _medium14_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def source() -> str:
        item = kube.get("deployment", "view-source")
        _labels_include(item, {"app": "view-source", "component": "source"})
        require(item.get("spec", {}).get("replicas") == 1, "view-source must have one replica")
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "view-source image differs")
        _mount_source(item, container, "/srv", "configMap", "view-source-page")
        _port(container, "source-http", 8080)
        _http_readiness(container, "source-http", 8080, period=5, failure=None)
        require(_ready_count(kube, "app=view-source") == 1, "view-source is not Ready")
        return "one Ready source replica serves the projected source page"

    def source_service() -> str:
        _service(kube, "view-source-svc", {"app": "view-source"}, 8080, "source-http", 8080)
        return _body_probe(kube, "aipc-eval-view-source", "http://view-source-svc:8080", "source=v1\n", labels={"access": "view"})

    def mirror() -> str:
        item = kube.get("deployment", "view-mirror")
        _labels_include(item, {"app": "view-mirror", "component": "mirror"})
        _rolling(item, replicas=2, min_ready=5, deadline=150)
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "view-mirror image differs")
        _secret_env(container, "TOKEN", "view-key", "TOKEN")
        _emptydir(item, container, "/www")
        _port(container, "view-http", 8081)
        _http_readiness(container, "view-http", 8081, period=5, failure=2)
        text = _command_has(container, "TOKEN", "mirror-ok", "view-source-svc", "/www/index.html", "rm", "sleep 2")
        _bounded_fetch(text, 5)
        require(_ready_count(kube, "app=view-mirror") == 2, "view-mirror does not have two Ready replicas")
        return "two Ready mirrors use the declared token and bounded failure-sensitive source loop"

    def view_service() -> str:
        _service(kube, "view-svc", {"app": "view-mirror"}, 80, "view-http", 8081)
        return _body_probe(kube, "aipc-eval-view-body", "http://view-svc", "view:source=v1\n", labels={"access": "view"})

    def hardening() -> str:
        return _hardened(kube.get("deployment", "view-source"), "view-source", set()) + "; " + _hardened(kube.get("deployment", "view-mirror"), "view-mirror", {"/www"})

    def ingress() -> str:
        return _live_ingress(
            kube,
            [
                ("aipc-eval-source-mirror", "http://view-source-svc:8080", {"component": "mirror"}),
                ("aipc-eval-source-view", "http://view-source-svc:8080", {"access": "view"}),
                ("aipc-eval-mirror-view", "http://view-svc", {"access": "view"}),
            ],
            [
                ("aipc-eval-source-deny", "http://view-source-svc:8080"),
                ("aipc-eval-mirror-deny", "http://view-svc"),
            ],
        )

    def update() -> str:
        source_container = _sole_container(kube.get("deployment", "view-source"))["name"]
        mirror_container = _sole_container(kube.get("deployment", "view-mirror"))["name"]

        def bodies(value: str) -> bool:
            return (
                _all_pod_bodies(kube, "app=view-source", source_container, 8080, value, 1)
                and _all_pod_bodies(kube, "app=view-mirror", mirror_container, 8081, f"view:{value}", 2)
            )

        return _live_update(
            kube,
            "configmap",
            "view-source-page",
            "index.html",
            "source=v2\n",
            "source=v1\n",
            bodies,
            timeout=150,
            controllers=[("deployment", "view-source"), ("deployment", "view-mirror")],
            selectors=["app=view-source", "app=view-mirror"],
            changed_check=lambda: _body_probe(kube, "aipc-eval-view-v2", "http://view-svc", "view:source=v2\n", labels={"access": "view"}) and None,
            restored_check=lambda: _body_probe(kube, "aipc-eval-view-v1", "http://view-svc", "view:source=v1\n", labels={"access": "view"}) and None,
        )

    def outage() -> str:
        saved = _snapshot(kube, "service", "view-source-svc")
        try:
            kube.run(["delete", "service", "view-source-svc", "-n", kube.namespace, "--wait=true"])
            kube.wait_until(
                lambda: _ready_count(kube, "app=view-mirror") == 0 and _endpoint_count(kube, "view-svc") == 0,
                "mirror readiness loss after source Service outage",
                60,
                2,
            )
        finally:
            _restore_object(kube, saved, "view-source-svc")
            kube.wait_until(
                lambda: _endpoint_count(kube, "view-source-svc") == 1 and _endpoint_count(kube, "view-svc") == 2,
                "source and mirror endpoint recovery",
                150,
                2,
            )
        _body_probe(kube, "aipc-eval-view-recovered", "http://view-svc", "view:source=v1\n", labels={"access": "view"})
        return "source Service outage withdrew mirrors and exact restoration recovered every endpoint"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-view-ns", candidate_path)),
        ("R02", "mutable source page", lambda: _exact_configmap(kube, "view-source-page", {"index.html": "source=v1\n"}, immutable=False)),
        ("R03", "immutable mirror key", lambda: _exact_secret(kube, "view-key", "TOKEN", "mirror-ok", immutable=True)),
        ("R04", "source Deployment", source),
        ("R05", "source Service", source_service),
        ("R06", "failure-sensitive mirror Deployment", mirror),
        ("R07", "mirror Service exact response", view_service),
        ("R08", "two-tier pod hardening", hardening),
        ("R09", "two-tier live ingress matrix", ingress),
        ("R10", "source-to-mirror live propagation", update),
        ("R11", "reversible source Service outage", outage),
    ]


def _medium15_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def claim() -> str:
        item = kube.get("pvc", "artifact-store")
        assert_claim(item, "64Mi")
        require(item.get("spec", {}).get("volumeMode", "Filesystem") == "Filesystem", "artifact-store volumeMode differs")
        require(item.get("status", {}).get("phase") == "Bound", "artifact-store is not Bound")
        return "artifact-store is an exact Bound 64Mi ReadWriteOnce Filesystem claim"

    def seed() -> str:
        workload, container, result = _job_contract(kube, "artifact-seed", 60, "artifact-seed-ok\n")
        _mount_source(workload, container, "/seed", "configMap", "artifact-seed-content")
        _mount_source(workload, container, "/data", "persistentVolumeClaim", "artifact-store")
        text = _command_has(container, "/seed/index.html", "/data/index.html", "artifact-seed-ok", "cp")
        require(
            re.search(
                r"(?:!\s+-[ef]|if\s+\[\s*!|\|\|\s*cp\b|\bcp\s+(?:--no-clobber|-\w*n\w*)\b)",
                text,
            )
            is not None,
            "seed program does not preserve an existing destination",
        )
        return result

    def deployment() -> str:
        item = kube.get("deployment", "artifact-web")
        _labels_include(item, {"app": "artifact-web"})
        require(item.get("spec", {}).get("replicas") == 1, "artifact-web must have one replica")
        require(item.get("spec", {}).get("strategy", {}).get("type") == "Recreate", "artifact-web strategy must be Recreate")
        _token_policy(item, False)
        init = _sole_container(item, init=True)
        _mount_source(item, init, "/data", "persistentVolumeClaim", "artifact-store")
        init_text = _command_has(init, "/data/index.html", "sleep 2")
        _bounded_attempts(init_text, 30, 2)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "artifact-web image differs")
        _secret_env(container, "CHECK", "artifact-check", "CHECK")
        _mount_source(item, container, "/data", "persistentVolumeClaim", "artifact-store")
        _port(container, "artifact-http", 8080)
        _http_readiness(container, "artifact-http", 8080, period=5, failure=None)
        _command_has(container, "CHECK", "serve", "/data")
        require(_ready_count(kube, "app=artifact-web") == 1, "artifact-web is not Ready")
        return "artifact-web has a bounded storage gate and one Ready Recreate server"

    def service() -> str:
        _service(kube, "artifact-web-svc", {"app": "artifact-web"}, 80, "artifact-http", 8080)
        return _body_probe(kube, "aipc-eval-artifact-body", "http://artifact-web-svc", "artifact=v1\n", labels={})

    def hardening() -> str:
        job = _job_workload(kube.get("job", "artifact-seed"))
        deployment_item = kube.get("deployment", "artifact-web")
        for container in [*all_containers(pod_spec(job)), *all_containers(pod_spec(deployment_item))]:
            _resource_contract(container)
        return _hardened(job, "artifact-seed", {"/data"}) + "; " + _hardened(deployment_item, "artifact-web", {"/data"})

    def durable() -> str:
        pods = _ready_pods(kube, "app=artifact-web")
        require(len(pods) == 1, "durability check requires one Ready artifact pod")
        original = pods[0]
        old_uid = original["metadata"]["uid"]
        container = _sole_container(kube.get("deployment", "artifact-web"))["name"]
        kube.exec(original["metadata"]["name"], "printf 'artifact=v2\\n' > /data/index.html", container)
        try:
            kube.run(["delete", "pod", original["metadata"]["name"], "-n", kube.namespace, "--wait=true"])

            def replacement() -> bool:
                return any(
                    _ready(pod) and pod.get("metadata", {}).get("uid") not in {"", old_uid}
                    for pod in kube.list("pods", "app=artifact-web")
                )

            kube.wait_until(replacement, "different-UID Ready artifact replacement", 90, 1)
            _body_probe(kube, "aipc-eval-artifact-v2", "http://artifact-web-svc", "artifact=v2\n", labels={})
        finally:
            for pod in kube.list("pods", "app=artifact-web"):
                name = pod.get("metadata", {}).get("name", "")
                if name:
                    kube.run(
                        ["exec", "-n", kube.namespace, name, "-c", container, "--", "sh", "-c", "printf 'artifact=v1\\n' > /data/index.html"],
                        check=False,
                        timeout=15,
                    )
        return "the v2 artifact survived a different-UID replacement and the evaluator restored v1"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "medium-artifact-ns", candidate_path)),
        ("R02", "immutable artifact seed", lambda: _exact_configmap(kube, "artifact-seed-content", {"index.html": "artifact=v1\n"}, immutable=True)),
        ("R03", "immutable artifact check", lambda: _exact_secret(kube, "artifact-check", "CHECK", "serve", immutable=True)),
        ("R04", "artifact persistent claim", claim),
        ("R05", "idempotent seed Job", seed),
        ("R06", "bounded Recreate artifact server", deployment),
        ("R07", "initial artifact Service response", service),
        ("R08", "artifact availability budget", lambda: _pdb(kube, "artifact-web-pdb", {"app": "artifact-web"}, 1)),
        ("R09", "artifact hardening and resources", hardening),
        ("R10", "durable artifact replacement", durable),
    ]


PENDING_FACTORIES: dict[str, Callable[[Kubectl, Path | None], list[Check]]] = {
    "medium-006": _medium6_checks,
    "medium-007": _medium7_checks,
    "medium-008": _medium8_checks,
    "medium-009": _medium9_checks,
    "medium-010": _medium10_checks,
    "medium-011": _medium11_checks,
    "medium-012": _medium12_checks,
    "medium-013": _medium13_checks,
    "medium-014": _medium14_checks,
    "medium-015": _medium15_checks,
}


def pending_checks(
    task_id: str,
    kube: Kubectl,
    candidate_path: Path | None = None,
) -> list[Check]:
    try:
        checks = PENDING_FACTORIES[task_id](kube, candidate_path)
    except KeyError as exc:
        raise EvaluationInfrastructureError(f"pending oracle is not implemented for {task_id}") from exc
    identifiers = [item[0] for item in checks]
    require(set(identifiers) == PENDING_EXPECTED_REQUIREMENTS[task_id], "pending suite coverage is incomplete")
    require(len(identifiers) == len(set(identifiers)), "pending suite has duplicate requirement checks")
    return checks
