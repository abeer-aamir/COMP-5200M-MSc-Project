from __future__ import annotations

import base64
import hashlib
from typing import Any, Callable

from .core import (
    EvaluationError,
    Kubectl,
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
    raise EvaluationError(f"Service does not expose port {port}")


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
    raise EvaluationError(f"no {direction} NetworkPolicy selects role={role}")


def _pdb_for_role(kube: Kubectl, role: str) -> dict[str, Any]:
    for pdb in kube.list("poddisruptionbudgets"):
        labels = pdb.get("spec", {}).get("selector", {}).get("matchLabels", {})
        if labels.get("role") == role:
            return pdb
    raise EvaluationError(f"no PodDisruptionBudget selects role={role}")


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


def _check_namespace(kube: Kubectl, expected: str) -> str:
    namespace = kube.get("namespace", expected, namespace=False)
    require(namespace.get("metadata", {}).get("name") == expected, "namespace is absent")
    require(namespace.get("status", {}).get("phase") == "Active", "namespace is not Active")
    return f"namespace {expected} is Active"


def _task1_checks(kube: Kubectl) -> list[tuple[str, str, Callable[[], str | None]]]:
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
        require(base64.b64decode(encoded).decode() == "pilot-token", "API_KEY placeholder differs")
        spec = pod_spec(deployment)
        container = _main_container(deployment)
        refs = {
            item.get("configMapRef", {}).get("name")
            for item in container.get("envFrom", [])
        }
        require("api-config" in refs, "api-config is not exposed with envFrom")
        volume = _volume_for_mount(spec, container, "/etc/api/secrets")
        projected = volume.get("projected", {})
        require(projected.get("defaultMode") == 0o440, "projected Secret mode is not 0440")
        names = {
            source.get("secret", {}).get("name") for source in projected.get("sources", [])
        }
        require("api-credentials" in names, "api-credentials is not projected")
        for item in container.get("env", []):
            require(
                item.get("valueFrom", {}).get("secretKeyRef", {}).get("name") != "api-credentials",
                "api-credentials must not also be an environment variable",
            )
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
        require(int_or_string(port.get("targetPort")) == "8080", "api-svc targetPort is not 8080")
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
        ("R01", "isolated order-system namespace", lambda: _check_namespace(kube, "order-system")),
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


def _task2_checks(kube: Kubectl) -> list[tuple[str, str, Callable[[], str | None]]]:
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
        require(base64.b64decode(encoded).decode() == "pilot-token", "API_TOKEN placeholder differs")
        controller = kube.get("deployment", "controller")
        env_refs = [
            item.get("valueFrom", {}).get("secretKeyRef", {})
            for item in _main_container(controller).get("env", [])
        ]
        require(any(ref.get("name") == "pipeline-secret" and ref.get("key") == "API_TOKEN" for ref in env_refs), "controller API_TOKEN is not sourced from pipeline-secret")
        for workload in (
            controller,
            kube.get("statefulset", "worker"),
            kube.get("cronjob", "marker-job"),
        ):
            spec = _cron_spec(workload) if workload.get("kind") == "CronJob" else pod_spec(workload)
            _assert_no_secret_volume(spec, "pipeline-secret")
        return "API_TOKEN is controller-only environment data and is never mounted"

    def r04() -> str:
        service = kube.get("service", "worker-svc")
        require(service["spec"].get("clusterIP") == "None", "worker-svc is not headless")
        require(service["spec"].get("selector", {}).get("role") == "worker", "worker-svc does not select workers")
        port = _service_port(service, 8080)
        require(port.get("protocol", "TCP") == "TCP", "worker-svc port is not TCP")
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
        require("-ge 2" in text or ">= 2" in text, "controller readiness does not require at least two successes")
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
        ("R01", "isolated pipeline-ns namespace", lambda: _check_namespace(kube, "pipeline-ns")),
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


def run_suite(task_id: str, kube: Kubectl) -> dict[str, Any]:
    if task_id == "pilot-001":
        checks = _task1_checks(kube)
    elif task_id == "pilot-002":
        checks = _task2_checks(kube)
    else:
        raise EvaluationError(f"unknown task id {task_id!r}")
    actual_ids = {requirement_id for requirement_id, _, _ in checks}
    require(actual_ids == EXPECTED_REQUIREMENTS[task_id], "suite requirement coverage is incomplete")
    collector = ResultCollector(task_id)
    for requirement_id, name, operation in checks:
        collector.check(requirement_id, name, operation)
    return collector.report()
