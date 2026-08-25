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

Check = tuple[str, str, Callable[[], str | None]]


def _scope(kube: Kubectl, expected: str, candidate_path: Path | None) -> str:
    # Import lazily to avoid a module-import cycle: suites.py exposes the
    # long-standing, mutation-tested namespace/scope oracle and imports these
    # five factories only after defining its core imports.
    from .suites import _check_namespace

    if candidate_path is None:
        raise EvaluationInfrastructureError(
            "the private evaluator was not given the deployed candidate manifest"
        )
    return _check_namespace(kube, expected, candidate_path)


def _container(workload: dict[str, Any], name: str, *, init: bool = False) -> dict[str, Any]:
    key = "initContainers" if init else "containers"
    matches = [item for item in pod_spec(workload).get(key, []) if item.get("name") == name]
    require(len(matches) == 1, f"expected exactly one {key[:-1]} named {name}")
    return matches[0]


def _exact_configmap(kube: Kubectl, name: str, expected: dict[str, str], *, mutable: bool = True) -> str:
    item = kube.get("configmap", name)
    require(item.get("data") == expected, f"{name} data differs")
    require(not item.get("binaryData"), f"{name} has unexpected binaryData")
    if mutable:
        require(item.get("immutable") is not True, f"{name} must remain mutable")
    return f"{name} has the exact data contract"


def _secret(kube: Kubectl, name: str, key: str, value: str) -> str:
    item = kube.get("secret", name)
    require(item.get("type") == "Opaque", f"{name} must be Opaque")
    require(item.get("immutable") is True, f"{name} must be immutable")
    require(set(item.get("data", {})) == {key}, f"{name} key set differs")
    try:
        decoded = base64.b64decode(item["data"][key], validate=True).decode("utf-8")
    except Exception as exc:
        raise EvaluationInfrastructureError(f"could not decode {name}/{key}") from exc
    require(decoded == value, f"{name}/{key} differs")
    return f"{name} is exact and immutable"


def _claim(kube: Kubectl, name: str, size: str) -> str:
    claim = kube.get("pvc", name)
    assert_claim(claim, size)
    require(claim.get("status", {}).get("phase") == "Bound", f"{name} is not Bound")
    return f"{name} is an exact Bound ReadWriteOnce claim"


def _port(container: dict[str, Any], name: str, number: int) -> None:
    matches = [
        item
        for item in container.get("ports", [])
        if item.get("name") == name
        and item.get("containerPort") == number
        and item.get("protocol", "TCP") == "TCP"
    ]
    require(len(matches) == 1, f"named TCP port {name}:{number} differs")


def _probe(container: dict[str, Any], kind: str, path: str, port: str, period: int, failure: int) -> None:
    probe = container.get(kind, {})
    require(
        probe.get("httpGet", {}) == {"path": path, "port": port},
        f"{kind} HTTP target differs",
    )
    require(probe.get("periodSeconds", 10) == period, f"{kind} period differs")
    require(probe.get("failureThreshold", 3) == failure, f"{kind} failure threshold differs")


def _hardened(workload: dict[str, Any], label: str) -> None:
    spec = pod_spec(workload)
    security = spec.get("securityContext", {})
    require(
        {key: security.get(key) for key in ("runAsNonRoot", "runAsUser", "runAsGroup", "fsGroup")}
        == {"runAsNonRoot": True, "runAsUser": 1000, "runAsGroup": 1000, "fsGroup": 1000},
        f"{label} pod security context differs",
    )
    for container in all_containers(spec):
        context = container.get("securityContext", {})
        require(context.get("allowPrivilegeEscalation") is False, f"{label} permits privilege escalation")
        require(context.get("readOnlyRootFilesystem") is True, f"{label} root filesystem is writable")
        require(set(context.get("capabilities", {}).get("drop", [])) == {"ALL"}, f"{label} must drop ALL capabilities")


def _ready(pod: dict[str, Any]) -> bool:
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in pod.get("status", {}).get("conditions", [])
    )


def _ready_count(kube: Kubectl, selector: str) -> int:
    return sum(_ready(pod) for pod in kube.list("pods", selector))


def _endpoint_uids(kube: Kubectl, service: str) -> set[str]:
    return {
        address.get("targetRef", {}).get("uid", "")
        for subset in kube.get("endpoints", service).get("subsets", [])
        for address in subset.get("addresses", [])
        if address.get("targetRef", {}).get("uid")
    }


def _endpoint_count(kube: Kubectl, service: str) -> int:
    return len(_endpoint_uids(kube, service))


def _service(
    kube: Kubectl,
    name: str,
    selector: dict[str, str],
    port: int,
    target: str,
    *,
    headless: bool = False,
) -> dict[str, Any]:
    item = kube.get("service", name)
    spec = item.get("spec", {})
    require(spec.get("type", "ClusterIP") == "ClusterIP", f"{name} is not ClusterIP")
    require(spec.get("selector") == selector, f"{name} selector differs")
    if headless:
        require(spec.get("clusterIP") == "None", f"{name} is not headless")
        require(spec.get("publishNotReadyAddresses", False) is False, f"{name} publishes NotReady addresses")
    else:
        require(spec.get("clusterIP") not in {None, "", "None"}, f"{name} lacks a cluster IP")
    ports = [item for item in spec.get("ports", []) if item.get("port") == port]
    require(len(ports) == 1, f"{name} must expose exactly the required port")
    require(
        ports[0].get("targetPort") == target and ports[0].get("protocol", "TCP") == "TCP",
        f"{name} targetPort differs",
    )
    return item


def _body_probe(kube: Kubectl, name: str, url: str, expected: str, *, labels: dict[str, str]) -> str:
    encoded = base64.b64encode(expected.encode()).decode()
    return kube.run_probe_pod(
        name,
        f'test "$(wget -q -T 5 -O - {url} | base64 | tr -d \'\\n\')" = {encoded}',
        labels=labels,
        timeout_seconds=35,
    )


def _deny_probe(kube: Kubectl, name: str, url: str) -> None:
    kube.run_probe_pod(
        name,
        f"wget -q -T 4 -O /dev/null {url}",
        labels={},
        expect_success=False,
        timeout_seconds=20,
    )


def _all_pod_bodies(kube: Kubectl, selector: str, container: str, port: int, expected: str, count: int) -> bool:
    encoded = base64.b64encode(expected.encode()).decode()
    pods = [pod for pod in kube.list("pods", selector) if _ready(pod)]
    if len(pods) != count:
        return False
    for pod in pods:
        result = kube.run(
            [
                "exec", "-n", kube.namespace, pod["metadata"]["name"], "-c", container,
                "--", "sh", "-ec",
                f'test "$(wget -q -T 5 -O - http://127.0.0.1:{port}/ | base64 | tr -d \'\\n\')" = {encoded}',
            ],
            check=False,
        )
        if result.returncode:
            return False
    return True


def _rollout(kube: Kubectl, kind: str, name: str, selector: str, count: int, timeout: int) -> None:
    kube.rollout(kind, name, timeout)
    require(_ready_count(kube, selector) == count, f"{name} does not have {count} Ready pods")


def _deployment_controls(workload: dict[str, Any], *, replicas: int, min_ready: int, deadline: int) -> None:
    spec = workload.get("spec", {})
    require(spec.get("replicas") == replicas, "replica count differs")
    rolling = spec.get("strategy", {}).get("rollingUpdate", {})
    require(int_or_string(rolling.get("maxUnavailable")) == "0", "maxUnavailable must be 0")
    require(int_or_string(rolling.get("maxSurge")) == "1", "maxSurge must be 1")
    require(spec.get("minReadySeconds") == min_ready, "minReadySeconds differs")
    require(spec.get("progressDeadlineSeconds") == deadline, "progressDeadlineSeconds differs")


def _pdb(kube: Kubectl, name: str, selector: dict[str, str], minimum: int) -> None:
    spec = kube.get("pdb", name).get("spec", {})
    require(spec.get("selector", {}).get("matchLabels") == selector, f"{name} selector differs")
    require(int_or_string(spec.get("minAvailable")) == str(minimum), f"{name} minAvailable differs")
    require("maxUnavailable" not in spec, f"{name} must not set maxUnavailable")


def _limits(kube: Kubectl, limit_name: str, quota_name: str, expected: dict[str, str]) -> str:
    limits = kube.get("limitrange", limit_name).get("spec", {}).get("limits", [])
    require(
        limits == [{
            "type": "Container",
            "defaultRequest": {"cpu": "20m", "memory": "32Mi"},
            "default": {"cpu": "200m", "memory": "128Mi"},
        }],
        f"{limit_name} differs",
    )
    hard = kube.get("resourcequota", quota_name).get("spec", {}).get("hard", {})
    require(set(hard) == set(expected), f"{quota_name} key set differs")
    require({key: int_or_string(value) for key, value in hard.items()} == expected, f"{quota_name} values differ")
    return "LimitRange and ResourceQuota are exact"


def _role(
    kube: Kubectl,
    name: str,
    expected: set[tuple[str, str, str, str]],
) -> None:
    actual: set[tuple[str, str, str, str]] = set()
    role = kube.get("role", name)
    for rule in role.get("rules", []):
        groups = rule.get("apiGroups", [])
        resources = rule.get("resources", [])
        names = rule.get("resourceNames", [])
        verbs = rule.get("verbs", [])
        require(len(groups) == len(resources) == len(names) == len(verbs) == 1, f"{name} rules must be atomic and exact")
        actual.add((groups[0], resources[0], names[0], verbs[0]))
    require(actual == expected, f"{name} permissions differ")


def _binding(kube: Kubectl, name: str, role: str, service_account: str) -> None:
    item = kube.get("rolebinding", name)
    require(
        item.get("roleRef") == {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": role},
        f"{name} roleRef differs",
    )
    subjects = item.get("subjects", [])
    require(len(subjects) == 1, f"{name} must bind one subject")
    subject = subjects[0]
    require(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace", kube.namespace) == kube.namespace,
        f"{name} subject differs",
    )


def _policies(kube: Kubectl, expected: set[str]) -> dict[str, dict[str, Any]]:
    items = kube.list("networkpolicies")
    actual = {item.get("metadata", {}).get("name"): item for item in items}
    require(set(actual) == expected, "NetworkPolicy name set differs")
    return actual


def _selector(spec: dict[str, Any], *, labels: dict[str, str] | None = None, expression: tuple[str, set[str]] | None = None) -> None:
    selector = spec.get("podSelector", {})
    if labels is not None:
        require(selector.get("matchLabels") == labels, "NetworkPolicy selector differs")
        require(not selector.get("matchExpressions"), "NetworkPolicy has extra selector expressions")
    if expression is not None:
        key, values = expression
        require(
            selector.get("matchExpressions") == [{"key": key, "operator": "In", "values": list(values)}]
            or (
                len(selector.get("matchExpressions", [])) == 1
                and selector["matchExpressions"][0].get("key") == key
                and selector["matchExpressions"][0].get("operator") == "In"
                and set(selector["matchExpressions"][0].get("values", [])) == values
            ),
            "NetworkPolicy expression selector differs",
        )
        require(not selector.get("matchLabels"), "NetworkPolicy has extra selector labels")


def _ports(rule: dict[str, Any], expected: set[tuple[str, int]]) -> None:
    actual: set[tuple[str, int]] = set()
    for item in rule.get("ports", []):
        require(isinstance(item, dict), "NetworkPolicy port entry is malformed")
        value = item.get("port")
        require(
            isinstance(value, int)
            or (isinstance(value, str) and value.isdigit()),
            "NetworkPolicy must state the required numeric port",
        )
        actual.add((item.get("protocol", "TCP"), int(value)))
    require(actual == expected and len(rule.get("ports", [])) == len(expected), "NetworkPolicy ports differ")


def _peers(rule: dict[str, Any], direction: str, expected: set[tuple[str, str]]) -> None:
    actual: set[tuple[str, str]] = set()
    peers = rule.get(direction, [])
    for peer in peers:
        labels = peer.get("podSelector", {}).get("matchLabels", {}) if isinstance(peer, dict) else {}
        require(len(labels) == 1 and set(peer) == {"podSelector"}, "NetworkPolicy peer must be one same-namespace pod label")
        actual.add(next(iter(labels.items())))
    require(actual == expected and len(peers) == len(expected), "NetworkPolicy peers differ")


def _ingress(policy: dict[str, Any], selector: dict[str, str], peers: set[tuple[str, str]], port: int) -> None:
    spec = policy.get("spec", {})
    _selector(spec, labels=selector)
    require(set(spec.get("policyTypes", ["Ingress"])) == {"Ingress"}, "ingress policyTypes differ")
    rules = spec.get("ingress", [])
    require(len(rules) == 1, "ingress policy must have exactly one rule")
    _peers(rules[0], "from", peers)
    _ports(rules[0], {("TCP", port)})


def _isolated(policy: dict[str, Any], labels: dict[str, str]) -> None:
    spec = policy.get("spec", {})
    _selector(spec, labels=labels)
    require(set(spec.get("policyTypes", ["Ingress"])) == {"Ingress"}, "isolation policyTypes differ")
    require(spec.get("ingress", []) == [], "isolation policy must deny all ingress")


def _egress(policy: dict[str, Any], selector: dict[str, str], destination: dict[str, str], port: int) -> None:
    spec = policy.get("spec", {})
    _selector(spec, labels=selector)
    require(set(spec.get("policyTypes", ["Egress"])) == {"Egress"}, "egress policyTypes differ")
    rules = spec.get("egress", [])
    require(len(rules) == 2, "egress policy must have dependency and DNS rules")
    dependency = [rule for rule in rules if any(item.get("port") == port for item in rule.get("ports", []))]
    dns = [rule for rule in rules if {item.get("protocol", "TCP") for item in rule.get("ports", [])} == {"UDP", "TCP"} and {item.get("port") for item in rule.get("ports", [])} == {53}]
    require(len(dependency) == len(dns) == 1, "egress dependency or DNS rule differs")
    _peers(dependency[0], "to", set(destination.items()))
    _ports(dependency[0], {("TCP", port)})
    _ports(dns[0], {("UDP", 53), ("TCP", 53)})


def _mount_source(workload: dict[str, Any], container: dict[str, Any], path: str, source_kind: str, source_name: str, *, read_only: bool | None = None) -> None:
    mount = find_mount(container, path)
    require(mount is not None, f"mount {path} is absent")
    if read_only is not None:
        require(mount.get("readOnly", False) is read_only, f"mount {path} readOnly differs")
    volume = find_volume(pod_spec(workload), mount.get("name", ""))
    require(volume is not None, f"volume for {path} is absent")
    source = volume.get(source_kind, {})
    require(source.get("name", source.get("claimName")) == source_name, f"source for {path} differs")


def _secret_env(container: dict[str, Any], env: str, secret: str, key: str) -> None:
    matches = [item for item in container.get("env", []) if item.get("name") == env]
    require(len(matches) == 1, f"environment variable {env} differs")
    require(matches[0].get("valueFrom", {}).get("secretKeyRef") == {"name": secret, "key": key}, f"{env} secretKeyRef differs")


def _job_shape(kube: Kubectl, name: str, container_name: str, deadline: int, log: str) -> str:
    job = kube.get("job", name)
    spec = job.get("spec", {})
    require(spec.get("backoffLimit") == 0 and spec.get("activeDeadlineSeconds") == deadline, f"{name} execution bounds differ")
    pod = spec.get("template", {}).get("spec", {})
    require(pod.get("restartPolicy") == "Never", f"{name} restartPolicy differs")
    containers = pod.get("containers", [])
    require(len(containers) == 1 and containers[0].get("name") == container_name and containers[0].get("image") == "busybox:1.36.1", f"{name} container differs")
    return _job_complete(kube, name, deadline, log)


def _job_complete(kube: Kubectl, name: str, timeout: int, expected_log: str) -> str:
    kube.wait_until(
        lambda: job_terminal_condition(kube.get("job", name)) is not None,
        f"{name} terminal state",
        timeout,
        1,
    )
    require(job_terminal_condition(kube.get("job", name)) == "Complete", f"{name} did not Complete")
    pods = kube.list("pods", f"job-name={name}")
    require(len(pods) == 1, f"{name} must have one pod")
    logs = kube.run(["logs", pods[0]["metadata"]["name"], "-n", kube.namespace]).stdout
    require(logs == expected_log, f"{name} logs differ")
    return f"{name} completed with exact logs"


def _cron_job(kube: Kubectl, cron: str, job: str, timeout: int, expected_log: str) -> str:
    require(job.startswith("aipc-eval-"), "evaluator Job name must use aipc-eval- prefix")
    try:
        kube.delete("job", job)
        try:
            kube.run(["create", "job", job, f"--from=cronjob/{cron}", "-n", kube.namespace])
        except KubectlError as exc:
            kube._raise_evaluator_creation_failure(exc, f"probe Job {job}")
        return _job_complete(kube, job, timeout, expected_log)
    finally:
        kube.delete("job", job)


def _cron_spec(item: dict[str, Any]) -> dict[str, Any]:
    return item["spec"]["jobTemplate"]["spec"]["template"]["spec"]


def _bounded(command: str, attempts: int, sleep: int = 2) -> None:
    require("exit 1" in command or "exit 2" in command, "retry loop does not fail non-zero")
    require(re.search(rf"(?:-ge|-gt)\s+{attempts}\b|seq\s+(?:1\s+)?{attempts}\b", command) is not None, f"retry bound {attempts} is absent")
    require(f"sleep {sleep}" in command, f"retry interval must be {sleep} seconds")


def _stateful_controls(workload: dict[str, Any], service: str, replicas: int) -> None:
    spec = workload.get("spec", {})
    require(spec.get("serviceName") == service and spec.get("replicas") == replicas, "StatefulSet identity differs")
    require(spec.get("podManagementPolicy") == "OrderedReady", "podManagementPolicy must be OrderedReady")
    require(spec.get("persistentVolumeClaimRetentionPolicy") == {"whenDeleted": "Retain", "whenScaled": "Retain"}, "PVC retention differs")
    update = spec.get("updateStrategy", {})
    require(update.get("type", "RollingUpdate") == "RollingUpdate" and update.get("rollingUpdate", {}).get("partition", 0) == 0, "StatefulSet update strategy differs")
    assert_claim(volume_claim_template(workload, "data"), "64Mi")


def _snapshot(kube: Kubectl, kind: str, name: str) -> dict[str, Any]:
    current = kube.get(kind, name)
    restored = {"apiVersion": current["apiVersion"], "kind": current["kind"]}
    restored["metadata"] = {"name": name, "namespace": kube.namespace}
    for key in ("spec", "roleRef", "subjects", "data", "binaryData", "type", "immutable"):
        if key in current:
            restored[key] = current[key]
    return restored


def _uids(kube: Kubectl, selector: str) -> set[str]:
    return {pod.get("metadata", {}).get("uid", "") for pod in kube.list("pods", selector)}


def _best_effort_remove(kube: Kubectl, pod: str, container: str, path: str) -> None:
    """Cleanup must not replace the requirement failure that triggered it."""

    kube.run(
        ["exec", "-n", kube.namespace, pod, "-c", container, "--", "sh", "-c", f"rm -f {path}"],
        check=False,
        timeout=min(kube.command_timeout, 15),
    )


def _generations(kube: Kubectl, resources: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    return {(kind, name): kube.get(kind, name).get("metadata", {}).get("generation", 0) for kind, name in resources}


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
    try:
        kube.run(["patch", "configmap", name, "-n", kube.namespace, "--type=merge", "-p", json.dumps({"data": {key: changed}})])
        kube.wait_until(lambda: assertions(changed), f"{name} live update", timeout, 2)
        require(_generations(kube, controllers) == generations, "controller generation changed during ConfigMap update")
        require({selector: _uids(kube, selector) for selector in selectors} == uids, "pod UID changed during ConfigMap update")
    finally:
        kube.run(["patch", "configmap", name, "-n", kube.namespace, "--type=merge", "-p", json.dumps({"data": {key: original}})])
        kube.wait_until(lambda: assertions(original), f"{name} restoration", timeout, 2)
    require(_generations(kube, controllers) == generations, "controller generation changed during restoration")
    require({selector: _uids(kube, selector) for selector in selectors} == uids, "pod UID changed during restoration")
    return f"{name} propagated in place and was restored without rollout"


def _replacement_continuity(
    kube: Kubectl,
    selector: str,
    service: str,
    url: str,
    expected: str,
    access: dict[str, str],
    timeout: int,
) -> str:
    pods = [pod for pod in kube.list("pods", selector) if _ready(pod)]
    require(len(pods) == 2, "continuity test requires two Ready replicas")
    victim = pods[0]
    old_uid = victim["metadata"]["uid"]
    kube.run(["delete", "pod", victim["metadata"]["name"], "-n", kube.namespace, "--wait=true"])
    kube.wait_until(lambda: old_uid not in _endpoint_uids(kube, service), "deleted endpoint removal", 30, 1)
    encoded = base64.b64encode(expected.encode()).decode()
    command = f'i=0; while [ "$i" -lt 12 ]; do test "$(wget -q -T 5 -O - {url} | base64 | tr -d \'\\n\')" = {encoded}; i=$((i+1)); sleep 1; done'
    kube.run_probe_pod(f"aipc-eval-{service}-continuity"[:63], command, labels=access, timeout_seconds=35)
    kube.wait_until(
        lambda: any(_ready(pod) and pod.get("metadata", {}).get("uid") not in {old_uid, ""} for pod in kube.list("pods", selector)),
        "different-UID replacement",
        timeout,
        1,
    )
    return "all twelve requests succeeded during a different-UID replacement"


def very_hard_1_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def r05() -> str:
        job = kube.get("job", "release-seed")
        container = _container({"spec": {"template": job["spec"]["template"]}}, "seed")
        require(container.get("image") == "busybox:1.36.1", "seed image differs")
        _mount_source({"spec": {"template": job["spec"]["template"]}}, container, "/state", "persistentVolumeClaim", "release-state")
        text = command_text(container)
        require("schema-1" in text and "release-seed-ok" in text and "/state/schema" in text, "seed program differs")
        return _job_shape(kube, "release-seed", "seed", 90, "release-seed-ok\n")

    def r06() -> str:
        dep = kube.get("deployment", "release-backend")
        require(dep.get("spec", {}).get("replicas") == 2 and pod_labels(dep) == {"app": "release-backend", "role": "backend"}, "backend replicas or labels differ")
        require(pod_spec(dep).get("automountServiceAccountToken") is False, "backend token automount must be false")
        init = _container(dep, "wait-seed", init=True)
        require(init.get("image") == "busybox:1.36.1", "wait-seed image differs")
        _mount_source(dep, init, "/state", "persistentVolumeClaim", "release-state", read_only=True)
        _bounded(command_text(init), 45)
        main = _container(dep, "backend")
        require(main.get("image") == "busybox:1.36.1", "backend image differs")
        _secret_env(main, "TOKEN", "release-token", "TOKEN")
        _mount_source(dep, main, "/srv", "configMap", "release-page", read_only=True)
        _mount_source(dep, main, "/state", "persistentVolumeClaim", "release-state", read_only=True)
        _port(main, "backend-http", 8080); _probe(main, "readinessProbe", "/", "backend-http", 5, 3)
        text = command_text(main)
        require("release-key" in text and "schema-1" in text and "httpd" in text and "8080" in text, "backend command does not enforce dependencies")
        _rollout(kube, "deployment", "release-backend", "role=backend", 2, 120)
        return "backend dependency gate, wiring, probes, and two Ready replicas are exact"

    def r07() -> str:
        _service(kube, "release-backend-svc", {"app": "release-backend"}, 8080, "backend-http")
        _body_probe(kube, "aipc-eval-release-backend", "http://release-backend-svc:8080/", "release=v1\n", labels={"role": "gateway"})
        return "backend Service returns exact v1 content to a gateway"

    def r08() -> str:
        for kind, name in (("deployment", "release-backend"), ("deployment", "release-gateway"), ("job", "release-seed")):
            _hardened(kube.get(kind, name), name)
        cron = kube.get("cronjob", "release-audit")
        _hardened({"spec": {"template": cron["spec"]["jobTemplate"]["spec"]["template"]}}, "release-audit")
        return "every application, init, seed, and audit container is hardened"

    def r09() -> str:
        dep = kube.get("deployment", "release-gateway")
        require(pod_labels(dep) == {"app": "release-gateway", "role": "gateway"}, "gateway labels differ")
        _deployment_controls(dep, replicas=2, min_ready=5, deadline=150)
        require(dep.get("spec", {}).get("revisionHistoryLimit") == 2, "gateway revisionHistoryLimit differs")
        require(pod_spec(dep).get("automountServiceAccountToken") is False, "gateway token automount must be false")
        init = _container(dep, "wait-backend", init=True)
        require(init.get("image") == "busybox:1.36.1" and "release-backend-svc:8080" in command_text(init) and "release=v1" in command_text(init), "wait-backend differs")
        _bounded(command_text(init), 60)
        main = _container(dep, "gateway")
        require(main.get("image") == "busybox:1.36.1", "gateway image differs")
        _port(main, "gateway-http", 8081); _probe(main, "readinessProbe", "/", "gateway-http", 5, 2)
        mount = find_mount(main, "/www"); require(mount is not None, "gateway /www mount missing")
        volume = find_volume(pod_spec(dep), mount["name"]); require(volume is not None and "emptyDir" in volume, "gateway /www is not emptyDir")
        text = command_text(main)
        require("release-backend-svc:8080" in text and "rm -f /www/index.html" in text and "sleep 2" in text and "httpd" in text, "gateway mirror loop differs")
        _rollout(kube, "deployment", "release-gateway", "role=gateway", 2, 150)
        return "gateway bounded dependency, mirror loop, rollout, and readiness are exact"

    def r10() -> str:
        _service(kube, "release-svc", {"app": "release-gateway"}, 80, "gateway-http")
        _body_probe(kube, "aipc-eval-release-client", "http://release-svc/", "release=v1\n", labels={"access": "release"})
        return _replacement_continuity(kube, "role=gateway", "release-svc", "http://release-svc/", "release=v1\n", {"access": "release"}, 75)

    def r11() -> str:
        _pdb(kube, "release-backend-pdb", {"app": "release-backend"}, 1)
        _pdb(kube, "release-gateway-pdb", {"app": "release-gateway"}, 1)
        return "both disruption budgets are exact"

    def r12() -> str:
        policies = _policies(kube, {"release-default-deny", "backend-from-gateway", "gateway-from-clients", "seed-isolation", "audit-isolation"})
        default = policies["release-default-deny"].get("spec", {})
        _selector(default, expression=("role", {"backend", "gateway"}))
        require(set(default.get("policyTypes", ["Ingress"])) == {"Ingress"} and default.get("ingress", []) == [], "release default deny differs")
        _ingress(policies["backend-from-gateway"], {"role": "backend"}, {("role", "gateway")}, 8080)
        _ingress(policies["gateway-from-clients"], {"role": "gateway"}, {("access", "release"), ("role", "release-audit")}, 8081)
        _isolated(policies["seed-isolation"], {"job-name": "release-seed"}); _isolated(policies["audit-isolation"], {"role": "release-audit"})
        _deny_probe(kube, "aipc-eval-release-backend-deny", "http://release-backend-svc:8080/")
        _deny_probe(kube, "aipc-eval-release-gateway-deny", "http://release-svc/")
        return "exact policy set allows intended peers and denies unlabelled clients"

    def r13() -> str:
        kube.get("serviceaccount", "release-auditor")
        _role(kube, "release-page-reader", {("", "configmaps", "release-page", "get")})
        _binding(kube, "release-page-reader-binding", "release-page-reader", "release-auditor")
        require(kube.auth_can_i("release-auditor", "get", "configmap/release-page"), "auditor cannot get release-page")
        require(not kube.auth_can_i("release-auditor", "list", "configmaps") and not kube.auth_can_i("release-auditor", "get", "secrets"), "auditor has excess access")
        return "auditor identity has only the exact named ConfigMap permission"

    def r14() -> str:
        cron = kube.get("cronjob", "release-audit"); spec = cron.get("spec", {}); pod = _cron_spec(cron)
        require((spec.get("schedule"), spec.get("suspend"), spec.get("concurrencyPolicy"), spec.get("successfulJobsHistoryLimit"), spec.get("failedJobsHistoryLimit")) == ("*/15 * * * *", True, "Forbid", 1, 1), "audit CronJob controls differ")
        require(pod.get("serviceAccountName") == "release-auditor" and pod.get("restartPolicy") == "Never" and spec["jobTemplate"]["spec"].get("activeDeadlineSeconds") == 150, "audit pod identity or bounds differ")
        require(cron["spec"]["jobTemplate"]["spec"]["template"].get("metadata", {}).get("labels", {}).get("role") == "release-audit", "audit role label missing")
        container = _container({"spec": {"template": cron["spec"]["jobTemplate"]["spec"]["template"]}}, "audit")
        text = command_text(container); require("release-svc" in text and "release=v1" in text and "release-audit-ok" in text, "audit program differs"); _bounded(text, 60)
        return "suspended audit CronJob has the exact identity, schedule, and bounded check"

    def r15() -> str:
        return _cron_job(kube, "release-audit", "aipc-eval-release-audit", 150, "release-audit-ok\n")

    def r16() -> str:
        return _limits(kube, "release-defaults", "release-quota", {"pods": "12", "persistentvolumeclaims": "2", "requests.storage": "256Mi", "services": "4", "count/jobs.batch": "4", "count/cronjobs.batch": "2", "requests.cpu": "2", "requests.memory": "1Gi", "limits.cpu": "4", "limits.memory": "2Gi"})

    def r17() -> str:
        def assertions(value: str) -> bool:
            return _all_pod_bodies(kube, "role=backend", "backend", 8080, value, 2) and _all_pod_bodies(kube, "role=gateway", "gateway", 8081, value, 2)
        return _update_configmap(kube, "release-page", "index.html", "release=v2\n", "release=v1\n", assertions, 150, [("deployment", "release-backend"), ("deployment", "release-gateway")], ["role=backend", "role=gateway"])

    return [
        ("R01", "isolated release namespace", lambda: _scope(kube, "very-hard-release-ns", candidate_path)),
        ("R02", "mutable release page", lambda: _exact_configmap(kube, "release-page", {"index.html": "release=v1\n"})),
        ("R03", "immutable release token", lambda: _secret(kube, "release-token", "TOKEN", "release-key")),
        ("R04", "release state claim", lambda: _claim(kube, "release-state", "64Mi")),
        ("R05", "idempotent release seed", r05), ("R06", "release backend", r06),
        ("R07", "backend Service", r07), ("R08", "release hardening", r08),
        ("R09", "release gateway", r09), ("R10", "gateway continuity", r10),
        ("R11", "release disruption budgets", r11), ("R12", "release network isolation", r12),
        ("R13", "release auditor RBAC", r13), ("R14", "release audit schedule", r14),
        ("R15", "release audit execution", r15), ("R16", "release resource controls", r16),
        ("R17", "in-place release update", r17),
    ]


def very_hard_2_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def r05() -> str:
        job = kube.get("job", "ledger-bootstrap")
        wrapped = {"spec": {"template": job["spec"]["template"]}}
        c = _container(wrapped, "bootstrap")
        require(c.get("image") == "busybox:1.36.1" and "/bootstrap/epoch" in command_text(c) and "ledger-bootstrap-ok" in command_text(c), "bootstrap program differs")
        _mount_source(wrapped, c, "/bootstrap", "persistentVolumeClaim", "ledger-bootstrap-state")
        return _job_shape(kube, "ledger-bootstrap", "bootstrap", 90, "ledger-bootstrap-ok\n")

    def r06() -> str:
        _service(kube, "ledger-headless", {"app": "ledger"}, 8080, "ledger-http", headless=True)
        return "ledger headless Service is exact"

    def r07() -> str:
        sts = kube.get("statefulset", "ledger"); _stateful_controls(sts, "ledger-headless", 3)
        require(pod_labels(sts).get("app") == "ledger" and pod_spec(sts).get("automountServiceAccountToken") is False, "ledger labels or token setting differ")
        c = _container(sts, "ledger"); require(c.get("image") == "busybox:1.36.1", "ledger image differs")
        _secret_env(c, "SUFFIX", "ledger-suffix", "SUFFIX"); _mount_source(sts, c, "/config", "configMap", "ledger-prefix", read_only=True); _mount_source(sts, c, "/bootstrap", "persistentVolumeClaim", "ledger-bootstrap-state", read_only=True)
        require(find_mount(c, "/data") is not None, "ledger data mount missing"); _port(c, "ledger-http", 8080); _probe(c, "readinessProbe", "/", "ledger-http", 5, 3)
        text = command_text(c); require("$HOSTNAME" in text and "SUFFIX" in text and "/bootstrap/epoch" in text and "index.new" in text and "sleep 2" in text, "ledger atomic writer differs")
        _rollout(kube, "statefulset", "ledger", "app=ledger", 3, 150)
        return "three-member durable ledger StatefulSet is exact and Ready"

    def r08() -> str:
        _hardened(kube.get("statefulset", "ledger"), "ledger"); _hardened(kube.get("deployment", "ledger-observer"), "ledger-observer"); _hardened(kube.get("job", "ledger-bootstrap"), "ledger-bootstrap")
        return "ledger, observer, and bootstrap are hardened"

    def r09() -> str:
        _service(kube, "ledger-svc", {"app": "ledger"}, 80, "ledger-http")
        require(_endpoint_count(kube, "ledger-svc") == 3 and _endpoint_count(kube, "ledger-headless") == 3, "ledger endpoint counts differ")
        for ordinal in range(3):
            _body_probe(kube, f"aipc-eval-ledger-{ordinal}", f"http://ledger-{ordinal}.ledger-headless:8080/", f"entry-ledger-{ordinal}-stable-7\n", labels={"access": "ledger"})
        return "all three stable ledger identities return exact content"

    def r10() -> str:
        dep = kube.get("deployment", "ledger-observer"); _deployment_controls(dep, replicas=2, min_ready=5, deadline=180)
        require(pod_labels(dep).get("app") == "ledger-observer" and pod_spec(dep).get("automountServiceAccountToken") is False, "observer labels or token setting differ")
        c = _container(dep, "observer"); require(c.get("image") == "busybox:1.36.1", "observer image differs")
        _port(c, "observer-http", 8081); _probe(c, "readinessProbe", "/", "observer-http", 5, 2)
        mount = find_mount(c, "/www"); require(mount is not None and "emptyDir" in (find_volume(pod_spec(dep), mount["name"]) or {}), "observer /www must be emptyDir")
        text = command_text(c); require(all(f"ledger-{n}.ledger-headless" in text for n in range(3)) and "ledger-count=3" in text and "rm -f /www/index.html" in text and "sleep 3" in text, "observer all-member loop differs")
        _rollout(kube, "deployment", "ledger-observer", "app=ledger-observer", 2, 180)
        return "observer verifies every stable member before becoming Ready"

    def r11() -> str:
        _service(kube, "ledger-observer-svc", {"app": "ledger-observer"}, 80, "observer-http")
        _body_probe(kube, "aipc-eval-ledger-observer", "http://ledger-observer-svc/", "ledger-count=3\n", labels={"access": "ledger"})
        return _replacement_continuity(kube, "app=ledger-observer", "ledger-observer-svc", "http://ledger-observer-svc/", "ledger-count=3\n", {"access": "ledger"}, 90)

    def r12() -> str:
        _pdb(kube, "ledger-pdb", {"app": "ledger"}, 2); _pdb(kube, "ledger-observer-pdb", {"app": "ledger-observer"}, 1)
        return "ledger disruption budgets are exact"

    def r13() -> str:
        policies = _policies(kube, {"ledger-default-deny", "ledger-from-observer", "observer-from-client", "bootstrap-isolation", "observer-egress"})
        default = policies["ledger-default-deny"].get("spec", {}); _selector(default, expression=("app", {"ledger", "ledger-observer"})); require(default.get("ingress", []) == [] and set(default.get("policyTypes", ["Ingress"])) == {"Ingress"}, "ledger default deny differs")
        _ingress(policies["ledger-from-observer"], {"app": "ledger"}, {("app", "ledger-observer"), ("access", "ledger")}, 8080)
        _ingress(policies["observer-from-client"], {"app": "ledger-observer"}, {("access", "ledger")}, 8081)
        _isolated(policies["bootstrap-isolation"], {"job-name": "ledger-bootstrap"}); _egress(policies["observer-egress"], {"app": "ledger-observer"}, {"app": "ledger"}, 8080)
        _deny_probe(kube, "aipc-eval-ledger-deny", "http://ledger-svc/"); _deny_probe(kube, "aipc-eval-observer-deny", "http://ledger-observer-svc/")
        return "ledger policy graph and live unlabelled denials are exact"

    def r14() -> str:
        kube.get("serviceaccount", "ledger-inspector")
        _role(kube, "ledger-status-reader", {("apps", "statefulsets", "ledger", "get"), ("apps", "statefulsets/status", "ledger", "get")})
        _binding(kube, "ledger-status-reader-binding", "ledger-status-reader", "ledger-inspector")
        require(pod_spec(kube.get("deployment", "ledger-observer")).get("serviceAccountName") == "ledger-inspector", "observer does not use ledger-inspector")
        require(kube.auth_can_i("ledger-inspector", "get", "statefulset/ledger") and not kube.auth_can_i("ledger-inspector", "list", "statefulsets") and not kube.auth_can_i("ledger-inspector", "get", "secrets"), "inspector live permissions differ")
        return "observer uses exact named-resource status RBAC"

    def r15() -> str:
        return _limits(kube, "ledger-defaults", "ledger-quota", {"pods": "14", "persistentvolumeclaims": "5", "requests.storage": "512Mi", "services": "4", "count/jobs.batch": "3", "requests.cpu": "2", "requests.memory": "1Gi", "limits.cpu": "4", "limits.memory": "2Gi"})

    def r16() -> str:
        old = kube.get("pod", "ledger-1")["metadata"]["uid"]
        kube.exec("ledger-1", "printf 'oracle-marker\\n' > /data/oracle-marker", "ledger")
        try:
            kube.run(["delete", "pod", "ledger-1", "-n", kube.namespace, "--wait=true"])
            kube.wait_until(lambda: (lambda p: _ready(p) and p.get("metadata", {}).get("uid") != old)(kube.get("pod", "ledger-1")), "different-UID ledger-1 replacement", 120, 1)
            kube.exec("ledger-1", 'test "$(cat /data/oracle-marker)" = oracle-marker', "ledger")
            kube.wait_until(lambda: _endpoint_count(kube, "ledger-svc") == 3 and _endpoint_count(kube, "ledger-headless") == 3, "ledger Service recovery", 150, 1)
        finally:
            _best_effort_remove(kube, "ledger-1", "ledger", "/data/oracle-marker")
        return "ledger-1 preserved evaluator data across replacement"

    def r17() -> str:
        try:
            kube.scale("statefulset", "ledger", 2)
            kube.wait_until(lambda: _ready_count(kube, "app=ledger") == 2 and _endpoint_count(kube, "ledger-svc") == 2 and _endpoint_count(kube, "ledger-headless") == 2 and _ready_count(kube, "app=ledger-observer") == 0 and _endpoint_count(kube, "ledger-observer-svc") == 0, "two-member degraded state", 120, 1)
        finally:
            kube.scale("statefulset", "ledger", 3)
            kube.wait_until(lambda: _ready_count(kube, "app=ledger") == 3 and _ready_count(kube, "app=ledger-observer") == 2 and _endpoint_count(kube, "ledger-svc") == 3 and _endpoint_count(kube, "ledger-headless") == 3 and _endpoint_count(kube, "ledger-observer-svc") == 2, "full ledger recovery", 180, 1)
        return "observer readiness and all Services tracked the 3-2-3 transition"

    return [
        ("R01", "isolated ledger namespace", lambda: _scope(kube, "very-hard-ledger-ns", candidate_path)),
        ("R02", "ledger prefix", lambda: _exact_configmap(kube, "ledger-prefix", {"PREFIX": "entry"})),
        ("R03", "ledger suffix", lambda: _secret(kube, "ledger-suffix", "SUFFIX", "stable")),
        ("R04", "bootstrap state", lambda: _claim(kube, "ledger-bootstrap-state", "32Mi")),
        ("R05", "ledger bootstrap", r05), ("R06", "ledger headless Service", r06),
        ("R07", "ledger StatefulSet", r07), ("R08", "ledger hardening", r08),
        ("R09", "stable ledger identities", r09), ("R10", "ledger observer", r10),
        ("R11", "observer continuity", r11), ("R12", "ledger disruption budgets", r12),
        ("R13", "ledger network isolation", r13), ("R14", "ledger inspector RBAC", r14),
        ("R15", "ledger resource controls", r15), ("R16", "ledger replacement persistence", r16),
        ("R17", "ledger scale transition", r17),
    ]


def very_hard_3_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def r04() -> str:
        ds = kube.get("daemonset", "node-agent")
        require(pod_labels(ds) == {"app": "node-agent", "component": "agent"}, "agent labels differ")
        spec = pod_spec(ds)
        require(spec.get("automountServiceAccountToken") is False, "agent token automount must be false")
        require(not spec.get("nodeSelector") and not spec.get("tolerations"), "agent must not constrain or broaden scheduling")
        c = _container(ds, "agent"); require(c.get("image") == "busybox:1.36.1", "agent image differs")
        _mount_source(ds, c, "/srv", "configMap", "node-status", read_only=True)
        _port(c, "agent-http", 8090); _probe(c, "startupProbe", "/", "agent-http", 5, 18); _probe(c, "readinessProbe", "/", "agent-http", 5, 2)
        require("httpd" in command_text(c) and "8090" in command_text(c) and "/srv" in command_text(c), "agent command differs")
        _rollout(kube, "daemonset", "node-agent", "component=agent", 1, 120)
        return "one exact Ready node agent runs on the single schedulable node"

    def r05() -> str:
        _service(kube, "node-agent-svc", {"app": "node-agent"}, 8090, "agent-http")
        _body_probe(kube, "aipc-eval-chain-agent", "http://node-agent-svc:8090/", "node-ok\n", labels={"component": "collector"})
        return "agent Service returns exact content to collectors"

    def r06() -> str:
        dep = kube.get("deployment", "status-collector"); _deployment_controls(dep, replicas=2, min_ready=5, deadline=180)
        require(pod_labels(dep) == {"app": "status-collector", "component": "collector"} and pod_spec(dep).get("automountServiceAccountToken") is False, "collector labels or token setting differ")
        init = _container(dep, "wait-agent", init=True); text = command_text(init)
        require(init.get("image") == "busybox:1.36.1" and "node-agent-svc:8090" in text and "node-ok" in text, "collector init gate differs"); _bounded(text, 60)
        c = _container(dep, "collector"); require(c.get("image") == "busybox:1.36.1", "collector image differs"); _secret_env(c, "TOKEN", "collector-token", "TOKEN")
        _port(c, "collector-http", 8080); _probe(c, "readinessProbe", "/", "collector-http", 5, 2)
        mount = find_mount(c, "/www"); require(mount is not None and "emptyDir" in (find_volume(pod_spec(dep), mount["name"]) or {}), "collector /www must be emptyDir")
        text = command_text(c); require("collector-key" in text and "node-agent-svc:8090" in text and "collector:%s" in text and "rm -f /www/index.html" in text and "sleep 2" in text, "collector mirror program differs")
        _rollout(kube, "deployment", "status-collector", "component=collector", 2, 180)
        return "collector dependency, token, mirror, and rollout controls are exact"

    def r07() -> str:
        _service(kube, "status-collector-svc", {"app": "status-collector"}, 8080, "collector-http")
        _body_probe(kube, "aipc-eval-chain-collector", "http://status-collector-svc:8080/", "collector:node-ok\n", labels={"component": "dashboard"})
        return "collector Service returns the exact transformed body"

    def r08() -> str:
        dep = kube.get("deployment", "status-dashboard"); _deployment_controls(dep, replicas=2, min_ready=5, deadline=210)
        require(pod_labels(dep) == {"app": "status-dashboard", "component": "dashboard"} and pod_spec(dep).get("automountServiceAccountToken") is False, "dashboard labels or token setting differ")
        init = _container(dep, "wait-collector", init=True); text = command_text(init)
        require(init.get("image") == "busybox:1.36.1" and "status-collector-svc:8080" in text and "collector:node-ok" in text, "dashboard init gate differs"); _bounded(text, 75)
        c = _container(dep, "dashboard"); require(c.get("image") == "busybox:1.36.1", "dashboard image differs")
        _port(c, "dashboard-http", 8081); _probe(c, "readinessProbe", "/", "dashboard-http", 5, 2)
        mount = find_mount(c, "/www"); require(mount is not None and "emptyDir" in (find_volume(pod_spec(dep), mount["name"]) or {}), "dashboard /www must be emptyDir")
        text = command_text(c); require("status-collector-svc:8080" in text and "dashboard:%s" in text and "rm -f /www/index.html" in text and "sleep 2" in text, "dashboard mirror program differs")
        _rollout(kube, "deployment", "status-dashboard", "component=dashboard", 2, 210)
        return "dashboard dependency, mirror, and rollout controls are exact"

    def r09() -> str:
        _service(kube, "status-dashboard-svc", {"app": "status-dashboard"}, 80, "dashboard-http")
        _body_probe(kube, "aipc-eval-chain-dashboard", "http://status-dashboard-svc/", "dashboard:collector:node-ok\n", labels={"access": "status"})
        return _replacement_continuity(kube, "component=dashboard", "status-dashboard-svc", "http://status-dashboard-svc/", "dashboard:collector:node-ok\n", {"access": "status"}, 90)

    def r10() -> str:
        for kind, name in (("daemonset", "node-agent"), ("deployment", "status-collector"), ("deployment", "status-dashboard")):
            _hardened(kube.get(kind, name), name)
        cron = kube.get("cronjob", "status-audit"); _hardened({"spec": {"template": cron["spec"]["jobTemplate"]["spec"]["template"]}}, "status-audit")
        return "all chain containers are hardened and writes use only emptyDirs"

    def r11() -> str:
        _pdb(kube, "collector-pdb", {"app": "status-collector"}, 1); _pdb(kube, "dashboard-pdb", {"app": "status-dashboard"}, 1)
        return "collector and dashboard disruption budgets are exact"

    def r12() -> str:
        policies = _policies(kube, {"chain-default-deny", "agent-ingress", "collector-ingress", "dashboard-ingress", "collector-egress", "dashboard-egress"})
        default = policies["chain-default-deny"].get("spec", {}); _selector(default, expression=("component", {"agent", "collector", "dashboard"})); require(set(default.get("policyTypes", [])) == {"Ingress", "Egress"} and default.get("ingress", []) == [] and default.get("egress", []) == [], "chain default deny differs")
        _ingress(policies["agent-ingress"], {"component": "agent"}, {("component", "collector"), ("access", "status")}, 8090)
        _ingress(policies["collector-ingress"], {"component": "collector"}, {("component", "dashboard"), ("access", "status")}, 8080)
        _ingress(policies["dashboard-ingress"], {"component": "dashboard"}, {("access", "status"), ("role", "status-audit")}, 8081)
        _egress(policies["collector-egress"], {"component": "collector"}, {"component": "agent"}, 8090); _egress(policies["dashboard-egress"], {"component": "dashboard"}, {"component": "collector"}, 8080)
        for name, url in (("agent", "http://node-agent-svc:8090/"), ("collector", "http://status-collector-svc:8080/"), ("dashboard", "http://status-dashboard-svc/")):
            _deny_probe(kube, f"aipc-eval-chain-{name}-deny", url)
        return "six-policy dependency graph and live denials are exact"

    def r13() -> str:
        kube.get("serviceaccount", "status-auditor")
        _role(kube, "node-agent-reader", {("apps", "daemonsets", "node-agent", "get"), ("apps", "daemonsets/status", "node-agent", "get")})
        _binding(kube, "node-agent-reader-binding", "node-agent-reader", "status-auditor")
        require(kube.auth_can_i("status-auditor", "get", "daemonset/node-agent") and not kube.auth_can_i("status-auditor", "list", "daemonsets") and not kube.auth_can_i("status-auditor", "get", "secrets"), "status auditor permissions differ")
        return "status auditor has only exact node-agent get permissions"

    def r14() -> str:
        cron = kube.get("cronjob", "status-audit"); spec = cron.get("spec", {}); pod = _cron_spec(cron)
        require((spec.get("schedule"), spec.get("suspend"), spec.get("concurrencyPolicy"), spec.get("successfulJobsHistoryLimit"), spec.get("failedJobsHistoryLimit")) == ("7,37 * * * *", True, "Forbid", 1, 1), "status-audit schedule differs")
        require(pod.get("serviceAccountName") == "status-auditor" and pod.get("restartPolicy") == "Never" and spec["jobTemplate"]["spec"].get("activeDeadlineSeconds") == 210, "status-audit identity or bounds differ")
        c = _container({"spec": {"template": spec["jobTemplate"]["spec"]["template"]}}, "audit"); text = command_text(c); require("status-dashboard-svc" in text and "dashboard:collector:node-ok" in text and "status-audit-ok" in text, "status audit program differs"); _bounded(text, 90)
        return _cron_job(kube, "status-audit", "aipc-eval-status-audit", 210, "status-audit-ok\n")

    def r15() -> str:
        return _limits(kube, "chain-defaults", "chain-quota", {"pods": "14", "services": "5", "count/daemonsets.apps": "2", "count/deployments.apps": "4", "count/jobs.batch": "3", "count/cronjobs.batch": "2", "requests.cpu": "2", "requests.memory": "1Gi", "limits.cpu": "4", "limits.memory": "2Gi"})

    def r16() -> str:
        def assertions(value: str) -> bool:
            leaf = value.rstrip("\n")
            return _all_pod_bodies(kube, "component=agent", "agent", 8090, value, 1) and _all_pod_bodies(kube, "component=collector", "collector", 8080, f"collector:{leaf}\n", 2) and _all_pod_bodies(kube, "component=dashboard", "dashboard", 8081, f"dashboard:collector:{leaf}\n", 2)
        return _update_configmap(kube, "node-status", "index.html", "node-v2\n", "node-ok\n", assertions, 180, [("daemonset", "node-agent"), ("deployment", "status-collector"), ("deployment", "status-dashboard")], ["component=agent", "component=collector", "component=dashboard"])

    def r17() -> str:
        restored = _snapshot(kube, "service", "node-agent-svc")
        kube.run(["delete", "service", "node-agent-svc", "-n", kube.namespace, "--wait=true"])
        try:
            kube.wait_until(lambda: _ready_count(kube, "component=collector") == 0 and _ready_count(kube, "component=dashboard") == 0 and _endpoint_count(kube, "status-collector-svc") == 0 and _endpoint_count(kube, "status-dashboard-svc") == 0, "downstream chain outage", 120, 1)
        finally:
            kube.run(["apply", "-f", "-"], input_text=json.dumps(restored))
            kube.wait_until(lambda: _endpoint_count(kube, "node-agent-svc") == 1 and _endpoint_count(kube, "status-collector-svc") == 2 and _endpoint_count(kube, "status-dashboard-svc") == 2, "complete chain recovery", 210, 1)
        _body_probe(kube, "aipc-eval-chain-recovered", "http://status-dashboard-svc/", "dashboard:collector:node-ok\n", labels={"access": "status"})
        return "deleting and exactly restoring the agent Service drove outage and recovery"

    return [
        ("R01", "isolated status-chain namespace", lambda: _scope(kube, "very-hard-chain-ns", candidate_path)),
        ("R02", "mutable node status", lambda: _exact_configmap(kube, "node-status", {"index.html": "node-ok\n"})),
        ("R03", "immutable collector token", lambda: _secret(kube, "collector-token", "TOKEN", "collector-key")),
        ("R04", "node agent", r04), ("R05", "node agent Service", r05),
        ("R06", "status collector", r06), ("R07", "collector Service", r07),
        ("R08", "status dashboard", r08), ("R09", "dashboard continuity", r09),
        ("R10", "chain hardening", r10), ("R11", "chain disruption budgets", r11),
        ("R12", "chain network isolation", r12), ("R13", "status auditor RBAC", r13),
        ("R14", "status audit execution", r14), ("R15", "chain resource controls", r15),
        ("R16", "in-place chain update", r16), ("R17", "dependency Service outage", r17),
    ]


def very_hard_4_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def r04() -> str:
        account = kube.get("serviceaccount", "publisher-reader")
        require(account.get("automountServiceAccountToken") is True, "publisher-reader must explicitly enable token automount")
        return "publisher-reader explicitly receives an API token"

    def r05() -> str:
        _role(kube, "published-content-reader", {("", "configmaps", "published-content", "get")})
        require(kube.auth_can_i("publisher-reader", "get", "configmap/published-content"), "publisher-reader cannot get published-content")
        require(not kube.auth_can_i("publisher-reader", "list", "configmaps") and not kube.auth_can_i("publisher-reader", "get", "secrets"), "publisher-reader has excess access")
        return "publisher-reader Role grants only one named ConfigMap get"

    def r06() -> str:
        _binding(kube, "published-content-reader-binding", "published-content-reader", "publisher-reader")
        return "publisher-reader is the binding's only subject"

    def r07() -> str:
        dep = kube.get("deployment", "authorized-publisher"); _deployment_controls(dep, replicas=2, min_ready=5, deadline=180)
        require(pod_labels(dep) == {"app": "authorized-publisher", "component": "publisher"} and pod_spec(dep).get("serviceAccountName") == "publisher-reader", "publisher identity differs")
        c = _container(dep, "publisher"); require(c.get("image") == "busybox:1.36.1", "publisher image differs"); _secret_env(c, "TOKEN", "publisher-token", "TOKEN")
        _mount_source(dep, c, "/srv", "configMap", "published-content", read_only=True)
        mount = find_mount(c, "/work"); require(mount is not None and "emptyDir" in (find_volume(pod_spec(dep), mount["name"]) or {}), "publisher /work must be emptyDir")
        _port(c, "publisher-http", 8080)
        text = command_text(c)
        require("/api/v1/namespaces/" in text and "/configmaps/published-content" in text and "Authorization: Bearer" in text and "/serviceaccount/token" in text and "wget" in text and "-T 5" in text and "--no-check-certificate" in text, "publisher does not perform the required bounded authenticated named API request")
        require("publish-key" in text and "/work/authorized" in text and "rm -f /work/authorized" in text and "sleep 3" in text and "httpd" in text, "publisher authorization loop differs")
        readiness = c.get("readinessProbe", {}); probe_text = " ".join(readiness.get("exec", {}).get("command", []))
        require(readiness.get("periodSeconds", 10) == 5 and readiness.get("failureThreshold", 3) == 2 and "/work/authorized" in probe_text and "127.0.0.1:8080" in probe_text and "/srv/index.html" in probe_text, "publisher readiness does not bind API authorization to current body")
        _rollout(kube, "deployment", "authorized-publisher", "component=publisher", 2, 180)
        return "publishers are Ready only after bounded named API authorization and exact local content"

    def r08() -> str:
        _service(kube, "authorized-publisher-svc", {"app": "authorized-publisher"}, 8080, "publisher-http")
        _body_probe(kube, "aipc-eval-published-source", "http://authorized-publisher-svc:8080/", "published=v1\n", labels={"app": "authorized-consumer"})
        return "publisher Service returns exact content to consumers"

    def r09() -> str:
        dep = kube.get("deployment", "authorized-consumer"); _deployment_controls(dep, replicas=2, min_ready=5, deadline=210)
        require(pod_labels(dep) == {"app": "authorized-consumer", "component": "consumer"} and pod_spec(dep).get("automountServiceAccountToken") is False, "consumer labels or token setting differ")
        init = _container(dep, "wait-publisher", init=True); text = command_text(init)
        require(init.get("image") == "busybox:1.36.1" and "authorized-publisher-svc:8080" in text and "published=v1" in text, "consumer init gate differs"); _bounded(text, 75)
        c = _container(dep, "consumer"); _port(c, "consumer-http", 8081); _probe(c, "readinessProbe", "/", "consumer-http", 5, 2)
        mount = find_mount(c, "/www"); require(mount is not None and "emptyDir" in (find_volume(pod_spec(dep), mount["name"]) or {}), "consumer /www must be emptyDir")
        text = command_text(c); require("authorized-publisher-svc:8080" in text and "rm -f /www/index.html" in text and "sleep 2" in text and "httpd" in text, "consumer mirror program differs")
        _rollout(kube, "deployment", "authorized-consumer", "component=consumer", 2, 210)
        return "two consumers mirror authorized publisher content and are Ready"

    def r10() -> str:
        _service(kube, "authorized-consumer-svc", {"app": "authorized-consumer"}, 80, "consumer-http")
        _body_probe(kube, "aipc-eval-published-client", "http://authorized-consumer-svc/", "published=v1\n", labels={"access": "published"})
        return _replacement_continuity(kube, "component=consumer", "authorized-consumer-svc", "http://authorized-consumer-svc/", "published=v1\n", {"access": "published"}, 90)

    def r11() -> str:
        _hardened(kube.get("deployment", "authorized-publisher"), "authorized-publisher"); _hardened(kube.get("deployment", "authorized-consumer"), "authorized-consumer"); _hardened(kube.get("job", "authorization-audit"), "authorization-audit")
        require(pod_spec(kube.get("deployment", "authorized-consumer")).get("automountServiceAccountToken") is False, "consumer receives an API token")
        return "publisher, consumer, init, and audit hardening and token boundaries are exact"

    def r12() -> str:
        _pdb(kube, "publisher-pdb", {"app": "authorized-publisher"}, 1); _pdb(kube, "consumer-pdb", {"app": "authorized-consumer"}, 1)
        return "publisher and consumer disruption budgets are exact"

    def r13() -> str:
        policies = _policies(kube, {"rbac-default-deny", "publisher-ingress", "consumer-ingress", "audit-isolation"})
        default = policies["rbac-default-deny"].get("spec", {}); _selector(default, expression=("component", {"publisher", "consumer"})); require(default.get("ingress", []) == [] and set(default.get("policyTypes", ["Ingress"])) == {"Ingress"}, "RBAC default deny differs")
        _ingress(policies["publisher-ingress"], {"component": "publisher"}, {("component", "consumer"), ("access", "published")}, 8080)
        _ingress(policies["consumer-ingress"], {"component": "consumer"}, {("access", "published")}, 8081); _isolated(policies["audit-isolation"], {"job-name": "authorization-audit"})
        _deny_probe(kube, "aipc-eval-publisher-deny", "http://authorized-publisher-svc:8080/"); _deny_probe(kube, "aipc-eval-consumer-deny", "http://authorized-consumer-svc/")
        return "four exact policies isolate both Services from unlabelled clients"

    def r14() -> str:
        job = kube.get("job", "authorization-audit"); spec = job.get("spec", {}); pod = spec.get("template", {}).get("spec", {})
        require(spec.get("backoffLimit") == 0 and spec.get("activeDeadlineSeconds") == 210 and pod.get("restartPolicy") == "Never" and pod.get("serviceAccountName") == "publisher-reader", "authorization audit identity or bounds differ")
        require(spec.get("template", {}).get("metadata", {}).get("labels", {}).get("access") == "published", "authorization audit access label missing")
        c = _container({"spec": {"template": spec["template"]}}, "audit"); text = command_text(c)
        require("/configmaps/published-content" in text and "Authorization: Bearer" in text and "/serviceaccount/token" in text and "-T 5" in text and "--no-check-certificate" in text and "authorized-consumer-svc" in text and "authorization-audit-ok" in text, "authorization audit program differs"); _bounded(text, 90)
        return _job_complete(kube, "authorization-audit", 210, "authorization-audit-ok\n")

    def r15() -> str:
        return _limits(kube, "rbac-defaults", "rbac-quota", {"pods": "12", "services": "4", "configmaps": "3", "secrets": "3", "count/jobs.batch": "3", "requests.cpu": "2", "requests.memory": "1Gi", "limits.cpu": "4", "limits.memory": "2Gi"})

    def r16() -> str:
        def assertions(value: str) -> bool:
            return _all_pod_bodies(kube, "component=publisher", "publisher", 8080, value, 2) and _all_pod_bodies(kube, "component=consumer", "consumer", 8081, value, 2)
        return _update_configmap(kube, "published-content", "index.html", "published=v2\n", "published=v1\n", assertions, 180, [("deployment", "authorized-publisher"), ("deployment", "authorized-consumer")], ["component=publisher", "component=consumer"])

    def r17() -> str:
        restored = _snapshot(kube, "rolebinding", "published-content-reader-binding")
        kube.run(["delete", "rolebinding", "published-content-reader-binding", "-n", kube.namespace, "--wait=true"])
        try:
            kube.wait_until(lambda: _ready_count(kube, "component=publisher") == 0 and _ready_count(kube, "component=consumer") == 0 and _endpoint_count(kube, "authorized-publisher-svc") == 0 and _endpoint_count(kube, "authorized-consumer-svc") == 0, "authorization revocation propagation", 120, 1)
        finally:
            kube.run(["apply", "-f", "-"], input_text=json.dumps(restored))
            kube.wait_until(lambda: _ready_count(kube, "component=publisher") == 2 and _ready_count(kube, "component=consumer") == 2 and _endpoint_count(kube, "authorized-publisher-svc") == 2 and _endpoint_count(kube, "authorized-consumer-svc") == 2, "authorization restoration", 210, 1)
        _body_probe(kube, "aipc-eval-rbac-recovered", "http://authorized-consumer-svc/", "published=v1\n", labels={"access": "published"})
        return "RoleBinding revocation and exact restoration drove full readiness transition"

    return [
        ("R01", "isolated RBAC namespace", lambda: _scope(kube, "very-hard-rbac-ns", candidate_path)),
        ("R02", "mutable published content", lambda: _exact_configmap(kube, "published-content", {"index.html": "published=v1\n"})),
        ("R03", "immutable publisher token", lambda: _secret(kube, "publisher-token", "TOKEN", "publish-key")),
        ("R04", "publisher identity", r04), ("R05", "named ConfigMap Role", r05),
        ("R06", "publisher binding", r06), ("R07", "authorized publisher", r07),
        ("R08", "publisher Service", r08), ("R09", "authorized consumer", r09),
        ("R10", "consumer continuity", r10), ("R11", "RBAC chain hardening", r11),
        ("R12", "RBAC chain disruption budgets", r12), ("R13", "RBAC chain network isolation", r13),
        ("R14", "authorization audit", r14), ("R15", "RBAC resource controls", r15),
        ("R16", "in-place published-content update", r16), ("R17", "authorization revocation", r17),
    ]


def very_hard_5_checks(kube: Kubectl, candidate_path: Path | None = None) -> list[Check]:
    def r04() -> str:
        _service(kube, "archive-headless", {"app": "archive"}, 8080, "archive-http", headless=True)
        return "archive headless Service is exact"

    def r05() -> str:
        sts = kube.get("statefulset", "archive"); _stateful_controls(sts, "archive-headless", 3)
        require(pod_labels(sts) == {"app": "archive", "component": "archive"} and pod_spec(sts).get("automountServiceAccountToken") is False, "archive labels or token setting differ")
        init = _container(sts, "initialize", init=True); require(init.get("image") == "busybox:1.36.1" and "/data/generation" in command_text(init) and "g1" in command_text(init) and "if" in command_text(init), "idempotent initialize program differs")
        require(find_mount(init, "/data") is not None, "initialize does not mount data")
        c = _container(sts, "archive"); require(c.get("image") == "busybox:1.36.1", "archive image differs"); _secret_env(c, "SIGNING", "archive-signing", "SIGNING"); _mount_source(sts, c, "/config", "configMap", "archive-prefix", read_only=True)
        require(find_mount(c, "/data") is not None, "archive data mount missing"); _port(c, "archive-http", 8080); _probe(c, "readinessProbe", "/", "archive-http", 5, 3)
        text = command_text(c); require("$HOSTNAME" in text and "$SIGNING" in text and "/data/generation" in text and "index.new" in text and "sleep 3" in text, "archive atomic writer differs")
        _rollout(kube, "statefulset", "archive", "component=archive", 3, 180)
        return "three-member archive StatefulSet initializes and serves durable ordinal data"

    def r06() -> str:
        _service(kube, "archive-svc", {"app": "archive"}, 80, "archive-http")
        require(_endpoint_count(kube, "archive-svc") == 3 and _endpoint_count(kube, "archive-headless") == 3, "archive endpoint counts differ")
        for ordinal in range(3):
            _body_probe(kube, f"aipc-eval-archive-{ordinal}", f"http://archive-{ordinal}.archive-headless:8080/", f"archive-archive-{ordinal}-seal-g1\n", labels={"access": "archive"})
        return "all three archive ordinals return exact stable bodies"

    def r07() -> str:
        dep = kube.get("deployment", "archive-reader"); _deployment_controls(dep, replicas=2, min_ready=5, deadline=210)
        require(pod_labels(dep) == {"app": "archive-reader", "component": "reader"} and pod_spec(dep).get("automountServiceAccountToken") is False, "reader labels or token setting differ")
        c = _container(dep, "reader"); require(c.get("image") == "busybox:1.36.1", "reader image differs"); _mount_source(dep, c, "/config", "configMap", "archive-prefix", read_only=True)
        mount = find_mount(c, "/www"); require(mount is not None and "emptyDir" in (find_volume(pod_spec(dep), mount["name"]) or {}), "reader /www must be emptyDir")
        _port(c, "reader-http", 8081); _probe(c, "readinessProbe", "/", "reader-http", 5, 2)
        text = command_text(c); require(all(f"archive-{n}.archive-headless" in text for n in range(3)) and "seal-g1" in text and "ready=3" in text and "rm -f /www/index.html" in text and "sleep 3" in text, "reader all-member program differs")
        _rollout(kube, "deployment", "archive-reader", "component=reader", 2, 210)
        return "both readers require all current ordinal bodies before readiness"

    def r08() -> str:
        _service(kube, "archive-reader-svc", {"app": "archive-reader"}, 80, "reader-http")
        _body_probe(kube, "aipc-eval-archive-reader", "http://archive-reader-svc/", "archive-ready=3\n", labels={"access": "archive"})
        return "archive reader Service returns exact aggregate content"

    def r09() -> str:
        _hardened(kube.get("statefulset", "archive"), "archive"); _hardened(kube.get("deployment", "archive-reader"), "archive-reader")
        cron = kube.get("cronjob", "archive-compact"); _hardened({"spec": {"template": cron["spec"]["jobTemplate"]["spec"]["template"]}}, "archive-compact")
        return "archive, initialize, reader, and compact containers are hardened"

    def r10() -> str:
        policies = _policies(kube, {"archive-default-deny", "archive-ingress", "reader-ingress", "maintenance-isolation", "reader-egress", "maintenance-egress"})
        default = policies["archive-default-deny"].get("spec", {}); _selector(default, expression=("component", {"archive", "reader"})); require(set(default.get("policyTypes", [])) == {"Ingress", "Egress"} and default.get("ingress", []) == [] and default.get("egress", []) == [], "archive default deny differs")
        _ingress(policies["archive-ingress"], {"component": "archive"}, {("component", "reader"), ("role", "archive-maintenance"), ("access", "archive")}, 8080)
        _ingress(policies["reader-ingress"], {"component": "reader"}, {("role", "archive-maintenance"), ("access", "archive")}, 8081); _isolated(policies["maintenance-isolation"], {"role": "archive-maintenance"})
        _egress(policies["reader-egress"], {"component": "reader"}, {"component": "archive"}, 8080); _egress(policies["maintenance-egress"], {"role": "archive-maintenance"}, {"component": "reader"}, 8081)
        _deny_probe(kube, "aipc-eval-archive-deny", "http://archive-svc/"); _deny_probe(kube, "aipc-eval-reader-deny", "http://archive-reader-svc/")
        return "six exact policies isolate archive and reader Services"

    def r11() -> str:
        kube.get("serviceaccount", "archive-maintainer")
        _role(kube, "archive-maintenance-reader", {("apps", "statefulsets", "archive", "get"), ("", "configmaps", "archive-prefix", "get")})
        _binding(kube, "archive-maintenance-reader-binding", "archive-maintenance-reader", "archive-maintainer")
        require(kube.auth_can_i("archive-maintainer", "get", "statefulset/archive") and kube.auth_can_i("archive-maintainer", "get", "configmap/archive-prefix") and not kube.auth_can_i("archive-maintainer", "list", "statefulsets") and not kube.auth_can_i("archive-maintainer", "get", "secrets"), "maintainer permissions differ")
        return "maintainer has only the two exact named-resource get permissions"

    def r12() -> str:
        _pdb(kube, "archive-pdb", {"app": "archive"}, 2); _pdb(kube, "archive-reader-pdb", {"app": "archive-reader"}, 1)
        return "archive disruption budgets are exact"

    def r13() -> str:
        return _limits(kube, "archive-defaults", "archive-quota", {"pods": "14", "persistentvolumeclaims": "4", "requests.storage": "384Mi", "services": "4", "count/jobs.batch": "3", "count/cronjobs.batch": "2", "requests.cpu": "2", "requests.memory": "1Gi", "limits.cpu": "4", "limits.memory": "2Gi"})

    def r14() -> str:
        cron = kube.get("cronjob", "archive-compact"); spec = cron.get("spec", {}); pod = _cron_spec(cron)
        require((spec.get("schedule"), spec.get("suspend"), spec.get("concurrencyPolicy"), spec.get("successfulJobsHistoryLimit"), spec.get("failedJobsHistoryLimit")) == ("13 */2 * * *", True, "Forbid", 1, 1), "archive-compact schedule differs")
        require(pod.get("serviceAccountName") == "archive-maintainer" and pod.get("restartPolicy") == "Never" and spec["jobTemplate"]["spec"].get("activeDeadlineSeconds") == 210, "compact identity or bounds differ")
        c = _container({"spec": {"template": spec["jobTemplate"]["spec"]["template"]}}, "compact"); text = command_text(c)
        _mount_source({"spec": {"template": spec["jobTemplate"]["spec"]["template"]}}, c, "/data", "persistentVolumeClaim", "data-archive-0")
        require("archive-reader-svc" in text and "archive-ready=3" in text and "/data/compacted" in text and "archive-compact-ok" in text, "compact program differs"); _bounded(text, 90)
        try:
            result = _cron_job(kube, "archive-compact", "aipc-eval-archive-compact", 210, "archive-compact-ok\n")
            kube.exec("archive-0", 'test "$(cat /data/compacted)" = compacted', "archive")
            return result + "; marker is exact on archive-0"
        finally:
            _best_effort_remove(kube, "archive-0", "archive", "/data/compacted")

    def r15() -> str:
        def assertions(value: str) -> bool:
            prefix = value.rstrip("\n")
            archives = all(_all_pod_bodies(kube, f"statefulset.kubernetes.io/pod-name=archive-{n}", "archive", 8080, f"{prefix}-archive-{n}-seal-g1\n", 1) for n in range(3))
            return archives and _all_pod_bodies(kube, "component=reader", "reader", 8081, f"{prefix}-ready=3\n", 2)
        return _update_configmap(kube, "archive-prefix", "PREFIX", "archive2", "archive", assertions, 210, [("statefulset", "archive"), ("deployment", "archive-reader")], ["component=archive", "component=reader"])

    def r16() -> str:
        old = kube.get("pod", "archive-1")["metadata"]["uid"]
        kube.exec("archive-1", "printf 'survivor\\n' > /data/survivor", "archive")
        try:
            kube.run(["delete", "pod", "archive-1", "-n", kube.namespace, "--wait=true"])
            kube.wait_until(lambda: (lambda p: _ready(p) and p.get("metadata", {}).get("uid") != old)(kube.get("pod", "archive-1")), "different-UID archive-1 replacement", 150, 1)
            kube.exec("archive-1", 'test "$(cat /data/survivor)" = survivor; test "$(cat /data/generation)" = g1', "archive")
            kube.wait_until(lambda: _endpoint_count(kube, "archive-svc") == 3 and _endpoint_count(kube, "archive-reader-svc") == 2, "archive replacement recovery", 210, 1)
        finally:
            _best_effort_remove(kube, "archive-1", "archive", "/data/survivor")
        return "archive-1 retained survivor and generation across replacement"

    def r17() -> str:
        try:
            kube.scale("statefulset", "archive", 2)
            kube.wait_until(lambda: _ready_count(kube, "component=archive") == 2 and _endpoint_count(kube, "archive-svc") == 2 and _endpoint_count(kube, "archive-headless") == 2 and _ready_count(kube, "component=reader") == 0 and _endpoint_count(kube, "archive-reader-svc") == 0, "two-member archive state", 150, 1)
        finally:
            kube.scale("statefulset", "archive", 3)
            kube.wait_until(lambda: _ready_count(kube, "component=archive") == 3 and _ready_count(kube, "component=reader") == 2 and _endpoint_count(kube, "archive-svc") == 3 and _endpoint_count(kube, "archive-headless") == 3 and _endpoint_count(kube, "archive-reader-svc") == 2, "full archive recovery", 210, 1)
        _body_probe(kube, "aipc-eval-archive-recovered", "http://archive-reader-svc/", "archive-ready=3\n", labels={"access": "archive"})
        return "readers and all Services tracked the archive 3-2-3 transition"

    return [
        ("R01", "isolated maintenance namespace", lambda: _scope(kube, "very-hard-maintenance-ns", candidate_path)),
        ("R02", "mutable archive prefix", lambda: _exact_configmap(kube, "archive-prefix", {"PREFIX": "archive"})),
        ("R03", "immutable archive signing value", lambda: _secret(kube, "archive-signing", "SIGNING", "seal")),
        ("R04", "archive headless Service", r04), ("R05", "archive StatefulSet", r05),
        ("R06", "stable archive identities", r06), ("R07", "archive readers", r07),
        ("R08", "archive reader Service", r08), ("R09", "archive hardening", r09),
        ("R10", "archive network isolation", r10), ("R11", "archive maintainer RBAC", r11),
        ("R12", "archive disruption budgets", r12), ("R13", "archive resource controls", r13),
        ("R14", "archive compaction", r14), ("R15", "in-place archive prefix update", r15),
        ("R16", "archive replacement persistence", r16), ("R17", "archive scale transition", r17),
    ]
