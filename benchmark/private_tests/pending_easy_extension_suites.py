"""Hidden oracles for easy-006 through easy-015.

The historical ``pending_`` filename is retained for import stability, but the
suites are registered in the live dispatcher.  They score the
public contract, not a preferred reference manifest: extra harmless metadata,
equivalent shell syntax, and named or numeric Service targets are accepted when
their observable meaning is the same.  Source-only requirements are checked
against the exact deployed candidate file.  Every evaluator-created object is
deleted and every reversible candidate mutation has an unconditional restore.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Callable

import yaml

from .core import (
    EvaluationInfrastructureError,
    Kubectl,
    KubectlError,
    RequirementFailure,
    command_text,
    int_or_string,
    job_terminal_condition,
    pod_labels,
    pod_spec,
    require,
)
from .pending_medium_extension_suites import (
    _all_pod_bodies,
    _binding,
    _body_probe,
    _bounded_fetch,
    _config_env,
    _emptydir,
    _endpoint_count,
    _endpoint_uids,
    _exact_configmap,
    _exact_secret,
    _hardened,
    _http_readiness,
    _limit_range,
    _live_ingress,
    _live_update,
    _local_body,
    _mount_source,
    _pdb,
    _port,
    _quota,
    _ready,
    _ready_count,
    _ready_pods,
    _resource_contract,
    _rolling,
    _secret_env,
    _service,
    _sole_container,
    _token_policy,
    _writable_data_paths,
)
from .very_hard_suites import _scope


Check = tuple[str, str, Callable[[], str | None]]

PENDING_EXPECTED_REQUIREMENTS = {
    task_id: {f"R{number:02d}" for number in range(1, count + 1)}
    for task_id, count in {
        "easy-006": 7,
        "easy-007": 7,
        "easy-008": 7,
        "easy-009": 7,
        "easy-010": 7,
        "easy-011": 7,
        "easy-012": 7,
        "easy-013": 7,
        "easy-014": 7,
        "easy-015": 8,
    }.items()
}


def _candidate_objects(candidate_path: Path | None) -> list[dict[str, Any]]:
    if candidate_path is None:
        raise EvaluationInfrastructureError(
            "the pending evaluator was not given the deployed candidate manifest"
        )
    try:
        loaded = list(yaml.safe_load_all(candidate_path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationInfrastructureError(
            f"could not read deployed candidate manifest {candidate_path}"
        ) from exc

    objects: list[dict[str, Any]] = []
    for document in loaded:
        if document is None:
            continue
        if not isinstance(document, dict):
            raise EvaluationInfrastructureError("candidate YAML contains a non-object document")
        if document.get("kind") == "List":
            items = document.get("items", [])
            if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
                raise EvaluationInfrastructureError("candidate List contains malformed items")
            objects.extend(items)
        else:
            objects.append(document)
    return objects


def _source_object(candidate_path: Path | None, kind: str, name: str) -> dict[str, Any]:
    matches = [
        item
        for item in _candidate_objects(candidate_path)
        if str(item.get("kind", "")).lower() == kind.lower()
        and item.get("metadata", {}).get("name") == name
    ]
    require(len(matches) == 1, f"candidate source must contain exactly one {kind}/{name}")
    return matches[0]


def _source_container(candidate_path: Path | None, kind: str, name: str) -> dict[str, Any]:
    item = _source_object(candidate_path, kind, name)
    normalized = kind.lower()
    if normalized == "cronjob":
        template = item.get("spec", {}).get("jobTemplate", {}).get("spec", {}).get("template", {})
    else:
        template = item.get("spec", {}).get("template", {})
    containers = template.get("spec", {}).get("containers", []) or []
    require(len(containers) == 1, f"source {kind}/{name} must have one container")
    return containers[0]


def _source_omits_resources(candidate_path: Path | None, kind: str, name: str) -> str:
    resources = _source_container(candidate_path, kind, name).get("resources")
    require(
        resources is None or resources == {},
        f"source {kind}/{name} must omit explicit resource requests and limits",
    )
    return f"source {kind}/{name} leaves resource admission to the LimitRange"


def _exact_secret_values(
    kube: Kubectl,
    name: str,
    expected: dict[str, str],
    *,
    immutable: bool,
) -> str:
    item = kube.get("secret", name)
    require(item.get("type") == "Opaque", f"{name} must be Opaque")
    data = item.get("data", {})
    require(set(data) == set(expected), f"{name} key set differs")
    decoded: dict[str, str] = {}
    try:
        for key, value in data.items():
            decoded[key] = base64.b64decode(value, validate=True).decode("utf-8")
    except Exception as exc:
        raise RequirementFailure(f"{name} does not contain valid base64 UTF-8") from exc
    require(decoded == expected, f"{name} decoded values differ")
    require(item.get("immutable") is immutable, f"{name} immutable flag differs")
    return f"{name} is exact, Opaque, and {'immutable' if immutable else 'mutable'}"


def _command_references(container: dict[str, Any], tokens: tuple[str, ...]) -> str:
    text = command_text(container)
    for token in tokens:
        require(token in text, f"container program does not reference {token!r}")
    return text


def _failure_capable_program(container: dict[str, Any], tokens: tuple[str, ...]) -> str:
    text = _command_references(container, tokens)
    require(
        re.search(r"(?:\btest\b|\bcase\b|\bgrep\b|\[)", text) is not None,
        "container program does not validate its inputs",
    )
    return text


def _admitted_container(kube: Kubectl, selector: str) -> dict[str, Any]:
    pods = _ready_pods(kube, selector)
    require(pods, f"no Ready pod matches {selector}")
    containers = pods[0].get("spec", {}).get("containers", []) or []
    require(len(containers) == 1, f"admitted pod for {selector} must have one container")
    return containers[0]


def _strict_job_complete(
    kube: Kubectl,
    name: str,
    timeout: int,
    expected_log: str,
    *,
    require_defaults: bool = False,
) -> str:
    kube.wait_until(
        lambda: job_terminal_condition(kube.get("job", name)) is not None,
        f"Job {name} terminal condition",
        timeout,
        1,
    )
    job = kube.get("job", name)
    require(job_terminal_condition(job) == "Complete", f"{name} did not Complete")
    require(int(job.get("status", {}).get("failed", 0) or 0) == 0, f"{name} recorded a failed pod")
    pods = kube.list("pods", f"job-name={name}")
    require(len(pods) == 1, f"{name} must have exactly one pod")
    pod = pods[0]
    require(pod.get("status", {}).get("phase") != "Failed", f"{name} has a failed pod")
    containers = pod.get("spec", {}).get("containers", []) or []
    require(len(containers) == 1, f"{name} pod must have exactly one container")
    if require_defaults:
        _resource_contract(containers[0])
    logs = kube.run(["logs", pod["metadata"]["name"], "-n", kube.namespace]).stdout
    require(logs == expected_log, f"{name} logs differ")
    detail = " and admitted LimitRange defaults" if require_defaults else ""
    return f"{name} completed without a failed pod and emitted exact logs{detail}"


def _run_cron_strict(
    kube: Kubectl,
    cronjob: str,
    suffix: str,
    timeout: int,
    expected_log: str,
    *,
    require_defaults: bool = False,
) -> str:
    name = f"aipc-eval-{suffix}"
    try:
        kube.delete("job", name)
        try:
            kube.run(["create", "job", name, f"--from=cronjob/{cronjob}", "-n", kube.namespace])
        except KubectlError as exc:
            kube._raise_evaluator_creation_failure(exc, f"probe Job {name}")
        return _strict_job_complete(
            kube,
            name,
            timeout,
            expected_log,
            require_defaults=require_defaults,
        )
    finally:
        kube.delete("job", name)


def _cron_structure(kube: Kubectl, name: str, schedule: str, deadline: int) -> tuple[dict[str, Any], dict[str, Any]]:
    cron = kube.get("cronjob", name)
    spec = cron.get("spec", {})
    require(spec.get("schedule") == schedule, f"{name} schedule differs")
    require(spec.get("suspend") is True, f"{name} must be suspended")
    require(spec.get("concurrencyPolicy") == "Forbid", f"{name} concurrencyPolicy differs")
    require(spec.get("successfulJobsHistoryLimit") == 1, f"{name} successful history differs")
    require(spec.get("failedJobsHistoryLimit") == 1, f"{name} failed history differs")
    job_spec = spec.get("jobTemplate", {}).get("spec", {})
    require(job_spec.get("backoffLimit") == 0, f"{name} backoffLimit differs")
    require(job_spec.get("activeDeadlineSeconds") == deadline, f"{name} deadline differs")
    workload = {"spec": {"template": job_spec.get("template", {})}}
    require(pod_spec(workload).get("restartPolicy") == "Never", f"{name} restartPolicy differs")
    container = _sole_container(workload)
    require(container.get("image") == "busybox:1.36.1", f"{name} image differs")
    return workload, container


def _owned_jobs(kube: Kubectl, cronjob: str) -> list[dict[str, Any]]:
    return [
        job
        for job in kube.list("jobs")
        if any(
            owner.get("kind") == "CronJob" and owner.get("name") == cronjob
            for owner in job.get("metadata", {}).get("ownerReferences", []) or []
        )
    ]


def _service_account(kube: Kubectl, name: str) -> str:
    account = kube.get("serviceaccount", name)
    require(account.get("metadata", {}).get("name") == name, f"{name} is absent")
    require(
        account.get("automountServiceAccountToken") is not False,
        f"{name} disables API-token automount",
    )
    return f"{name} exists with API-token automount enabled"


def _exact_configmap_role(kube: Kubectl, name: str, resource_name: str) -> str:
    role = kube.get("role", name)
    actual: set[tuple[str, str, str, str]] = set()
    for rule in role.get("rules", []) or []:
        require(not rule.get("nonResourceURLs"), f"{name} has non-resource access")
        for group in rule.get("apiGroups", []) or []:
            for resource in rule.get("resources", []) or []:
                for named in rule.get("resourceNames", []) or ["*"]:
                    for verb in rule.get("verbs", []) or []:
                        actual.add((group, resource, named, verb))
    require(
        actual == {("", "configmaps", resource_name, "get")},
        f"{name} permissions differ",
    )
    return f"{name} grants only get on configmaps/{resource_name}"


def _api_read_program(container: dict[str, Any], resource: str, key: str, expected: str) -> str:
    text = command_text(container)
    lower = text.lower()
    require("authorization" in lower and "bearer" in lower, "API request lacks bearer authorization")
    require("serviceaccount" in lower and "token" in lower, "program does not use the mounted service-account token")
    require(
        "https://kubernetes.default.svc" in lower,
        "bearer token may only be sent to the in-cluster Kubernetes API",
    )
    require(
        "--no-check-certificate" in lower,
        "the frozen BusyBox wget requires the documented local TLS exception",
    )
    require(
        "/configmaps/" in text and resource in text,
        "program does not request exactly the named ConfigMap",
    )
    require(key in text and expected in text, "program does not validate the required ConfigMap value")
    _bounded_fetch(text, 5)
    require(
        re.search(r"(?:\btest\b|\bcase\b|\bgrep\b|\[)", text) is not None,
        "program does not reject the wrong ConfigMap value",
    )
    return text


def _eligible_node_names(kube: Kubectl) -> set[str]:
    eligible: set[str] = set()
    for node in kube.list("nodes", namespace=False):
        spec = node.get("spec", {})
        if spec.get("unschedulable") is True:
            continue
        blocking = any(
            taint.get("effect") in {"NoSchedule", "NoExecute"}
            for taint in spec.get("taints", []) or []
        )
        if not blocking:
            name = node.get("metadata", {}).get("name")
            if name:
                eligible.add(name)
    if not eligible:
        raise EvaluationInfrastructureError(
            "the frozen reference cluster has no node eligible for an untolerating DaemonSet"
        )
    return eligible


def _daemonset_nodes(kube: Kubectl, selector: str) -> set[str]:
    expected = _eligible_node_names(kube)
    pods = _ready_pods(kube, selector)
    actual = {pod.get("spec", {}).get("nodeName") for pod in pods}
    require(None not in actual, "a Ready DaemonSet pod has no nodeName")
    require(actual == expected and len(pods) == len(expected), "DaemonSet Ready-node coverage differs")
    return expected


def _daemonset_replacement(
    kube: Kubectl,
    selector: str,
    service: str,
    expected_body: str,
    labels: dict[str, str],
    timeout: int,
) -> str:
    pods = _ready_pods(kube, selector)
    require(pods, "replacement check requires a Ready DaemonSet pod")
    expected_count = len(pods)
    victim = pods[0]
    old_uid = victim.get("metadata", {}).get("uid", "")
    old_node = victim.get("spec", {}).get("nodeName", "")
    require(bool(old_uid and old_node), "replacement victim lacks UID or node identity")
    kube.run(["delete", "pod", victim["metadata"]["name"], "-n", kube.namespace, "--wait=true"])
    kube.wait_until(
        lambda: old_uid not in _endpoint_uids(kube, service),
        "deleted DaemonSet endpoint removal",
        30,
        1,
    )
    replacement: dict[str, Any] = {}

    def restored() -> bool:
        nonlocal replacement
        for pod in kube.list("pods", selector):
            if (
                _ready(pod)
                and pod.get("metadata", {}).get("uid") not in {None, "", old_uid}
                and pod.get("spec", {}).get("nodeName") == old_node
            ):
                replacement = pod
                return _endpoint_count(kube, service) == expected_count
        return False

    kube.wait_until(restored, "same-node different-UID DaemonSet replacement", timeout, 1)
    _body_probe(kube, f"aipc-eval-{service}-recovery"[:63], f"http://{service}", expected_body, labels=labels)
    return (
        f"{old_uid} was replaced on {old_node} by "
        f"{replacement.get('metadata', {}).get('uid')} and {expected_count} endpoints returned"
    )


def _pod_local_and_service(
    kube: Kubectl,
    workload_kind: str,
    workload_name: str,
    selector: str,
    service: str,
    port: int,
    expected: str,
    count: int,
    labels: dict[str, str],
) -> str:
    container = _sole_container(kube.get(workload_kind, workload_name))["name"]
    require(
        _all_pod_bodies(kube, selector, container, port, expected, count),
        "one or more Ready pods do not return the exact body",
    )
    _body_probe(kube, f"aipc-eval-{service}-body"[:63], f"http://{service}", expected, labels=labels)
    return f"all {count} Ready pods and {service} return the exact bytes"


def _easy6_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def deployment() -> str:
        item = kube.get("deployment", "maintenance-web")
        require(pod_labels(item).get("app") == "maintenance-web", "maintenance-web pod label differs")
        _rolling(item, replicas=2, min_ready=3, deadline=90)
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "maintenance-web image differs")
        _mount_source(item, container, "/srv", "configMap", "maintenance-page")
        _port(container, "maintenancehttp", 8080)
        _http_readiness(container, "maintenancehttp", 8080, period=5, failure=3)
        require(_ready_count(kube, "app=maintenance-web") == 2, "maintenance-web lacks two Ready replicas")
        return _hardened(item, "maintenance-web", _writable_data_paths(item))

    def update() -> str:
        container = _sole_container(kube.get("deployment", "maintenance-web"))["name"]

        def bodies(value: str) -> bool:
            return _all_pod_bodies(kube, "app=maintenance-web", container, 8080, value, 2)

        return _live_update(
            kube,
            "configmap",
            "maintenance-page",
            "index.html",
            "maintenance=v2\n",
            "maintenance=v1\n",
            bodies,
            timeout=120,
            controllers=[("deployment", "maintenance-web")],
            selectors=["app=maintenance-web"],
            changed_check=lambda: _body_probe(kube, "aipc-eval-maintenance-v2", "http://maintenance-svc", "maintenance=v2\n", labels={}) and None,
            restored_check=lambda: _body_probe(kube, "aipc-eval-maintenance-v1", "http://maintenance-svc", "maintenance=v1\n", labels={}) and None,
        )

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-maintenance-ns", candidate_path)),
        ("R02", "mutable maintenance page", lambda: _exact_configmap(kube, "maintenance-page", {"index.html": "maintenance=v1\n"}, immutable=False)),
        ("R03", "maintenance Deployment", deployment),
        ("R04", "maintenance Service", lambda: _service(kube, "maintenance-svc", {"app": "maintenance-web"}, 80, "maintenancehttp", 8080) and "maintenance-svc routes to the declared port"),
        ("R05", "maintenance availability budget", lambda: _pdb(kube, "maintenance-pdb", {"app": "maintenance-web"}, 1)),
        ("R06", "initial maintenance responses", lambda: _pod_local_and_service(kube, "deployment", "maintenance-web", "app=maintenance-web", "maintenance-svc", 8080, "maintenance=v1\n", 2, {})),
        ("R07", "live maintenance projection and restoration", update),
    ]


def _easy7_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def daemonset() -> str:
        item = kube.get("daemonset", "nodekey-web")
        require(pod_labels(item).get("app") == "nodekey-web", "nodekey-web pod label differs")
        spec = pod_spec(item)
        require(not spec.get("nodeSelector"), "nodekey-web must not use nodeSelector")
        require(not spec.get("tolerations"), "nodekey-web must not add tolerations")
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "nodekey-web image differs")
        _mount_source(item, container, "/srv", "secret", "nodekey-content", key_path=("DATA", "index.html"))
        _port(container, "nodekey-http", 8090)
        _http_readiness(container, "nodekey-http", 8090, period=5, failure=3)
        nodes = _daemonset_nodes(kube, "app=nodekey-web")
        return f"nodekey-web is Ready exactly once on each eligible node: {sorted(nodes)}"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-nodekey-ns", candidate_path)),
        ("R02", "immutable node-key Secret", lambda: _exact_secret(kube, "nodekey-content", "DATA", "node-key-ready\n", immutable=True)),
        ("R03", "node-key DaemonSet", daemonset),
        ("R04", "node-key Service", lambda: _service(kube, "nodekey-svc", {"app": "nodekey-web"}, 80, "nodekey-http", 8090) and "nodekey-svc routes to the declared port"),
        ("R05", "node-key ingress isolation", lambda: _live_ingress(kube, [("aipc-eval-nodekey-allow", "http://nodekey-svc", {"access": "nodekey"})], [("aipc-eval-nodekey-deny", "http://nodekey-svc")])),
        ("R06", "exact node-key responses", lambda: _pod_local_and_service(kube, "daemonset", "nodekey-web", "app=nodekey-web", "nodekey-svc", 8090, "node-key-ready\n", len(_eligible_node_names(kube)), {"access": "nodekey"})),
        ("R07", "node-key replacement", lambda: _daemonset_replacement(kube, "app=nodekey-web", "nodekey-svc", "node-key-ready\n", {"access": "nodekey"}, 90)),
    ]


def _easy8_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def cron() -> str:
        _cron_structure(kube, "pulse-audit", "17 */4 * * *", 60)
        return "pulse-audit has the declared suspended schedule and execution controls"

    def program() -> str:
        _, container = _cron_structure(kube, "pulse-audit", "17 */4 * * *", 60)
        _config_env(container, "PULSE", "pulse-settings", "PULSE")
        _secret_env(container, "MODE", "pulse-mode", "MODE")
        _failure_capable_program(container, ("PULSE", "MODE", "7", "audit", "pulse=7;mode=audit"))
        _source_omits_resources(candidate_path, "CronJob", "pulse-audit")
        return "pulse-audit validates both exact key references and leaves resources to admission"

    def execution() -> str:
        require(not _owned_jobs(kube, "pulse-audit"), "a normal Job appeared while pulse-audit was suspended")
        return _run_cron_strict(
            kube,
            "pulse-audit",
            "pulse-audit",
            60,
            "pulse=7;mode=audit\n",
            require_defaults=True,
        )

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-pulse-ns", candidate_path)),
        ("R02", "immutable pulse settings", lambda: _exact_configmap(kube, "pulse-settings", {"PULSE": "7"}, immutable=True)),
        ("R03", "immutable pulse mode", lambda: _exact_secret(kube, "pulse-mode", "MODE", "audit", immutable=True)),
        ("R04", "suspended pulse schedule", cron),
        ("R05", "pulse template inputs and program", program),
        ("R06", "pulse admission defaults", lambda: _limit_range(kube, "pulse-defaults")),
        ("R07", "executable pulse schedule", execution),
    ]


def _easy9_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def statefulset() -> str:
        item = kube.get("statefulset", "ordinal-web")
        spec = item.get("spec", {})
        require(spec.get("replicas") == 2, "ordinal-web replica count differs")
        require(spec.get("serviceName") == "ordinal-headless", "ordinal-web serviceName differs")
        require(spec.get("podManagementPolicy", "OrderedReady") == "OrderedReady", "pod management differs")
        update = spec.get("updateStrategy", {})
        require(update.get("type", "RollingUpdate") == "RollingUpdate", "update strategy differs")
        require(int(update.get("rollingUpdate", {}).get("partition", 0)) == 0, "rolling partition differs")
        require(pod_labels(item).get("app") == "ordinal-web", "ordinal-web pod label differs")
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "ordinal-web image differs")
        _config_env(container, "PREFIX", "ordinal-prefix", "PREFIX")
        _emptydir(item, container, "/www")
        _port(container, "ordinal-http", 8080)
        _http_readiness(container, "ordinal-http", 8080, period=5, failure=3)
        text = _command_references(container, ("PREFIX", "/www/index.html"))
        require(
            "HOSTNAME" in text or re.search(r"\bhostname\b", text) is not None,
            "container program does not derive the pod hostname",
        )
        require(_ready_count(kube, "app=ordinal-web") == 2, "ordinal-web lacks two Ready replicas")
        return _hardened(item, "ordinal-web", {"/www"})

    def stable_dns() -> str:
        encoded0 = base64.b64encode(b"member-ordinal-web-0\n").decode()
        encoded1 = base64.b64encode(b"member-ordinal-web-1\n").decode()
        command = (
            f'test "$(wget -q -T 5 -O - http://ordinal-web-0.ordinal-headless:8080/ | base64 | tr -d \'\\n\')" = {encoded0}; '
            f'test "$(wget -q -T 5 -O - http://ordinal-web-1.ordinal-headless:8080/ | base64 | tr -d \'\\n\')" = {encoded1}'
        )
        return kube.run_probe_pod("aipc-eval-ordinal-dns", command, labels={}, timeout_seconds=35)

    def replacement() -> str:
        original = kube.get("pod", "ordinal-web-0")
        old_uid = original.get("metadata", {}).get("uid", "")
        require(old_uid and _ready(original), "ordinal-web-0 is not initially Ready")
        kube.run(["delete", "pod", "ordinal-web-0", "-n", kube.namespace, "--wait=true"])
        kube.wait_until(lambda: old_uid not in _endpoint_uids(kube, "ordinal-headless"), "ordinal zero endpoint removal", 30, 1)

        def ready() -> bool:
            candidate = kube.get("pod", "ordinal-web-0")
            return (
                _ready(candidate)
                and candidate.get("metadata", {}).get("uid") not in {None, "", old_uid}
                and _endpoint_count(kube, "ordinal-headless") == 2
            )

        kube.wait_until(ready, "different-UID ordinal-web-0 replacement", 90, 1)
        container = _sole_container(kube.get("statefulset", "ordinal-web"))["name"]
        require(_local_body(kube, "ordinal-web-0", container, 8080, "member-ordinal-web-0\n"), "replacement ordinal body differs")
        stable_dns()
        return "ordinal-web-0 returned with a different UID, stable body, and two endpoints"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-ordinal-ns", candidate_path)),
        ("R02", "immutable ordinal prefix", lambda: _exact_configmap(kube, "ordinal-prefix", {"PREFIX": "member"}, immutable=True)),
        ("R03", "ordinal headless Service", lambda: _service(kube, "ordinal-headless", {"app": "ordinal-web"}, 8080, "ordinal-http", 8080, headless=True) and "ordinal-headless is correctly wired"),
        ("R04", "ordinal StatefulSet", statefulset),
        ("R05", "ordinal availability budget", lambda: _pdb(kube, "ordinal-pdb", {"app": "ordinal-web"}, 1)),
        ("R06", "stable ordinal DNS responses", stable_dns),
        ("R07", "stable ordinal replacement", replacement),
    ]


def _easy10_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def deployment() -> str:
        item = kube.get("deployment", "scale-web")
        require(item.get("spec", {}).get("replicas") == 1, "scale-web must start with one replica")
        require(pod_labels(item).get("app") == "scale-web", "scale-web pod label differs")
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "scale-web image differs")
        _mount_source(item, container, "/srv", "secret", "scale-page", key_path=("DATA", "index.html"))
        _port(container, "scale-http", 8080)
        _http_readiness(container, "scale-http", 8080, period=5, failure=3)
        _source_omits_resources(candidate_path, "Deployment", "scale-web")
        require(_ready_count(kube, "app=scale-web") == 1, "scale-web lacks one Ready replica")
        return "scale-web has the declared source template and one Ready replica"

    def initial() -> str:
        require(_endpoint_count(kube, "scale-svc") == 1, "scale-svc does not have one Ready endpoint")
        result = _pod_local_and_service(kube, "deployment", "scale-web", "app=scale-web", "scale-svc", 8080, "scale-ready\n", 1, {})
        _resource_contract(_admitted_container(kube, "app=scale-web"))
        return result + " with exact admission defaults"

    def scale() -> str:
        try:
            kube.scale("deployment", "scale-web", 2)
            kube.wait_until(
                lambda: _ready_count(kube, "app=scale-web") == 2 and _endpoint_count(kube, "scale-svc") == 2,
                "two Ready scale-web endpoints",
                60,
                1,
            )
            _pod_local_and_service(kube, "deployment", "scale-web", "app=scale-web", "scale-svc", 8080, "scale-ready\n", 2, {})
        finally:
            kube.scale("deployment", "scale-web", 1)
            kube.wait_until(
                lambda: _ready_count(kube, "app=scale-web") == 1 and _endpoint_count(kube, "scale-svc") == 1,
                "restored single scale-web endpoint",
                60,
                1,
            )
        _pod_local_and_service(kube, "deployment", "scale-web", "app=scale-web", "scale-svc", 8080, "scale-ready\n", 1, {})
        return "scale-web served exact bytes at two replicas and after unconditional restoration to one"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-scale-ns", candidate_path)),
        ("R02", "immutable scale page", lambda: _exact_secret(kube, "scale-page", "DATA", "scale-ready\n", immutable=True)),
        ("R03", "scale Deployment", deployment),
        ("R04", "scale Service", lambda: _service(kube, "scale-svc", {"app": "scale-web"}, 80, "scale-http", 8080) and "scale-svc routes to scale-web"),
        ("R05", "scale admission defaults", lambda: _limit_range(kube, "scale-defaults")),
        ("R06", "initial scale endpoint", initial),
        ("R07", "reversible scale transition", scale),
    ]


def _job_structure(kube: Kubectl, name: str, deadline: int) -> tuple[dict[str, Any], dict[str, Any]]:
    job = kube.get("job", name)
    spec = job.get("spec", {})
    require(spec.get("backoffLimit") == 0, f"{name} backoffLimit differs")
    require(spec.get("activeDeadlineSeconds") == deadline, f"{name} deadline differs")
    require(spec.get("completions", 1) == 1 and spec.get("parallelism", 1) == 1, f"{name} execution shape differs")
    workload = {"spec": {"template": spec.get("template", {})}}
    require(pod_spec(workload).get("restartPolicy") == "Never", f"{name} restartPolicy differs")
    container = _sole_container(workload)
    require(container.get("image") == "busybox:1.36.1", f"{name} image differs")
    return workload, container


def _easy11_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def job() -> str:
        workload, container = _job_structure(kube, "document-check", 60)
        spec = pod_spec(workload)
        require(spec.get("serviceAccountName") == "document-reader", "document-check uses the wrong ServiceAccount")
        require(spec.get("automountServiceAccountToken") is not False, "document-check disables API-token automount")
        _api_read_program(container, "access-document", "DATA", "authorized-read")
        return "document-check has a bounded authenticated named-ConfigMap API program"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-document-ns", candidate_path)),
        ("R02", "immutable access document", lambda: _exact_configmap(kube, "access-document", {"DATA": "authorized-read"}, immutable=True)),
        ("R03", "document reader identity", lambda: _service_account(kube, "document-reader")),
        ("R04", "document reader Role", lambda: _exact_configmap_role(kube, "document-reader-role", "access-document")),
        ("R05", "document reader binding", lambda: _binding(kube, "document-reader-binding", "document-reader-role", "document-reader")),
        ("R06", "document-check API program", job),
        ("R07", "document-check execution", lambda: _strict_job_complete(kube, "document-check", 60, "document-access-ok\n")),
    ]


def _easy12_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def deployment() -> str:
        item = kube.get("deployment", "greeting-web")
        require(pod_labels(item).get("app") == "greeting-web", "greeting-web pod label differs")
        _rolling(item, replicas=2, min_ready=3, deadline=90)
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "greeting-web image differs")
        _mount_source(item, container, "/srv", "configMap", "greeting-page")
        _port(container, "greeting-http", 8080)
        _http_readiness(container, "greeting-http", 8080, period=5, failure=3)
        require(_ready_count(kube, "app=greeting-web") == 2, "greeting-web lacks two Ready replicas")
        return "greeting-web has the declared mount, serving, probe, token, and rollout controls"

    def response() -> str:
        require(_endpoint_count(kube, "greeting-svc") == 2, "greeting-svc does not have two Ready addresses")
        return _body_probe(kube, "aipc-eval-greeting-body", "http://greeting-svc", "hello=easy\n", labels={"access": "greeting"})

    def replacement() -> str:
        result = _replacement_continuity_local(
            kube,
            "app=greeting-web",
            "greeting-svc",
            "hello=easy\n",
            {"access": "greeting"},
            60,
        )
        require(_endpoint_count(kube, "greeting-svc") == 2, "greeting-svc did not restore two endpoints")
        return result

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-greeting-ns", candidate_path)),
        ("R02", "immutable greeting page", lambda: _exact_configmap(kube, "greeting-page", {"index.html": "hello=easy\n"}, immutable=True)),
        ("R03", "greeting Deployment", deployment),
        ("R04", "greeting Service", lambda: _service(kube, "greeting-svc", {"app": "greeting-web"}, 80, "greeting-http", 8080) and "greeting-svc routes to greeting-web"),
        ("R05", "greeting ingress isolation", lambda: _live_ingress(kube, [("aipc-eval-greeting-allow", "http://greeting-svc", {"access": "greeting"})], [("aipc-eval-greeting-deny", "http://greeting-svc")])),
        ("R06", "greeting response and endpoints", response),
        ("R07", "greeting replacement continuity", replacement),
    ]


def _replacement_continuity_local(
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
    old_uid = victim.get("metadata", {}).get("uid", "")
    kube.run(["delete", "pod", victim["metadata"]["name"], "-n", kube.namespace, "--wait=true"])
    kube.wait_until(lambda: old_uid not in _endpoint_uids(kube, service), "deleted endpoint removal", 30, 1)
    encoded = base64.b64encode(expected.encode()).decode()
    command = (
        'i=0; while [ "$i" -lt 10 ]; do '
        f'test "$(wget -q -T 1 -O - http://{service} | base64 | tr -d \'\\n\')" = {encoded}; '
        'i=$((i+1)); sleep 1; done'
    )
    kube.run_probe_pod(f"aipc-eval-{service}-continuity"[:63], command, labels=labels, timeout_seconds=25)
    kube.wait_until(
        lambda: any(
            _ready(pod) and pod.get("metadata", {}).get("uid") not in {None, "", old_uid}
            for pod in kube.list("pods", selector)
        ),
        "different-UID Ready replacement",
        timeout,
        1,
    )
    return "all ten one-second-bounded exact requests succeeded during replacement"


def _easy13_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def controls() -> str:
        _job_structure(kube, "pair-build", 45)
        return "pair-build has one completion, one-way parallelism, no retry, and a 45-second deadline"

    def program() -> str:
        _, container = _job_structure(kube, "pair-build", 45)
        _secret_env(container, "LEFT", "pair-input", "LEFT")
        _secret_env(container, "RIGHT", "pair-input", "RIGHT")
        _failure_capable_program(container, ("LEFT", "RIGHT", "alpha", "beta"))
        _source_omits_resources(candidate_path, "Job", "pair-build")
        return "pair-build validates exact Secret inputs and leaves resources to admission"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-pair-ns", candidate_path)),
        ("R02", "immutable pair input", lambda: _exact_secret_values(kube, "pair-input", {"LEFT": "alpha", "RIGHT": "beta"}, immutable=True)),
        ("R03", "pair quota", lambda: _quota(kube, "pair-quota", {"pods": "4", "count/jobs.batch": "3", "requests.cpu": "500m", "requests.memory": "256Mi", "limits.cpu": "1", "limits.memory": "512Mi"})),
        ("R04", "pair defaults", lambda: _limit_range(kube, "pair-defaults")),
        ("R05", "pair Job controls", controls),
        ("R06", "pair program and source resources", program),
        ("R07", "pair Job execution", lambda: _strict_job_complete(kube, "pair-build", 45, "pair=alpha-beta\n", require_defaults=True)),
    ]


def _easy14_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def daemonset() -> str:
        item = kube.get("daemonset", "nodepage-web")
        require(pod_labels(item).get("app") == "nodepage-web", "nodepage-web pod label differs")
        spec = pod_spec(item)
        require(not spec.get("nodeSelector"), "nodepage-web must not use nodeSelector")
        require(not spec.get("tolerations"), "nodepage-web must not add tolerations")
        _token_policy(item, False)
        container = _sole_container(item)
        require(container.get("image") == "busybox:1.36.1", "nodepage-web image differs")
        _mount_source(item, container, "/srv", "configMap", "nodepage-content")
        _port(container, "nodepage-http", 8091)
        _http_readiness(container, "nodepage-http", 8091, period=5, failure=3)
        nodes = _daemonset_nodes(kube, "app=nodepage-web")
        return f"nodepage-web is Ready exactly once on each eligible node: {sorted(nodes)}"

    def initial() -> str:
        return _pod_local_and_service(
            kube,
            "daemonset",
            "nodepage-web",
            "app=nodepage-web",
            "nodepage-svc",
            8091,
            "nodepage=v1\n",
            len(_eligible_node_names(kube)),
            {"access": "nodepage"},
        )

    def update() -> str:
        container = _sole_container(kube.get("daemonset", "nodepage-web"))["name"]
        count = len(_eligible_node_names(kube))

        def bodies(value: str) -> bool:
            return _all_pod_bodies(kube, "app=nodepage-web", container, 8091, value, count)

        return _live_update(
            kube,
            "configmap",
            "nodepage-content",
            "index.html",
            "nodepage=v2\n",
            "nodepage=v1\n",
            bodies,
            timeout=120,
            controllers=[("daemonset", "nodepage-web")],
            selectors=["app=nodepage-web"],
            changed_check=lambda: _body_probe(kube, "aipc-eval-nodepage-v2", "http://nodepage-svc", "nodepage=v2\n", labels={"access": "nodepage"}) and None,
            restored_check=lambda: _body_probe(kube, "aipc-eval-nodepage-v1", "http://nodepage-svc", "nodepage=v1\n", labels={"access": "nodepage"}) and None,
        )

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-nodepage-ns", candidate_path)),
        ("R02", "mutable node page", lambda: _exact_configmap(kube, "nodepage-content", {"index.html": "nodepage=v1\n"}, immutable=False)),
        ("R03", "node page DaemonSet", daemonset),
        ("R04", "node page Service", lambda: _service(kube, "nodepage-svc", {"app": "nodepage-web"}, 80, "nodepage-http", 8091) and "nodepage-svc routes to nodepage-web"),
        ("R05", "node page ingress isolation", lambda: _live_ingress(kube, [("aipc-eval-nodepage-allow", "http://nodepage-svc", {"access": "nodepage"})], [("aipc-eval-nodepage-deny", "http://nodepage-svc")])),
        ("R06", "initial node page responses", initial),
        ("R07", "live node page projection and restoration", update),
    ]


def _easy15_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def cron() -> str:
        workload, _ = _cron_structure(kube, "schedule-audit", "29 */6 * * *", 60)
        spec = pod_spec(workload)
        require(spec.get("serviceAccountName") == "schedule-reader", "schedule-audit uses the wrong ServiceAccount")
        require(spec.get("automountServiceAccountToken") is not False, "schedule-audit disables API-token automount")
        return "schedule-audit has the declared schedule, identity, and execution controls"

    def program() -> str:
        _, container = _cron_structure(kube, "schedule-audit", "29 */6 * * *", 60)
        _api_read_program(container, "schedule-note", "NOTE", "checked")
        return "schedule-audit has a bounded authenticated named-ConfigMap API program"

    return [
        ("R01", "namespace and candidate scope", lambda: _scope(kube, "easy-schedule-ns", candidate_path)),
        ("R02", "immutable schedule note", lambda: _exact_configmap(kube, "schedule-note", {"NOTE": "checked"}, immutable=True)),
        ("R03", "schedule reader identity", lambda: _service_account(kube, "schedule-reader")),
        ("R04", "schedule note Role", lambda: _exact_configmap_role(kube, "schedule-note-reader", "schedule-note")),
        ("R05", "schedule note binding", lambda: _binding(kube, "schedule-note-reader-binding", "schedule-note-reader", "schedule-reader")),
        ("R06", "suspended schedule controls", cron),
        ("R07", "scheduled API program", program),
        ("R08", "executable scheduled audit", lambda: _run_cron_strict(kube, "schedule-audit", "schedule-audit", 60, "schedule-audit-ok\n")),
    ]


PENDING_FACTORIES: dict[str, Callable[[Kubectl, Path | None], list[Check]]] = {
    "easy-006": _easy6_checks,
    "easy-007": _easy7_checks,
    "easy-008": _easy8_checks,
    "easy-009": _easy9_checks,
    "easy-010": _easy10_checks,
    "easy-011": _easy11_checks,
    "easy-012": _easy12_checks,
    "easy-013": _easy13_checks,
    "easy-014": _easy14_checks,
    "easy-015": _easy15_checks,
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
