"""Headless Kubernetes adaptation of the AIPyCraft execution loop.

Keep the public pipeline export lazy.  The private benchmark evaluator imports
``aipycraft_k8s.commands``; eagerly importing the pipeline here made that safe
submodule import recurse through ``post_verifier`` and back into the partially
initialised evaluator package.  Normal test discovery happened to hide the
cycle, while importing the evaluator directly failed.
"""

from typing import Any


__all__ = ["KubernetesAIPyCraftPipeline"]


def __getattr__(name: str) -> Any:
    if name == "KubernetesAIPyCraftPipeline":
        from .pipeline import KubernetesAIPyCraftPipeline

        return KubernetesAIPyCraftPipeline
    raise AttributeError(name)
