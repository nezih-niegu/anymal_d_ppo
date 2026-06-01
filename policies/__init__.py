"""
policies/
=========
Modular policy wrappers for the ANYmal D evaluator.

Public registry
---------------
All concrete policies are available via ``POLICY_REGISTRY``:

    from policies import POLICY_REGISTRY
    policy = POLICY_REGISTRY["ppo_multivariate"]()
    policy.load("path/to/checkpoint.pt")
    action = policy.act(obs)

Stubs (diffusion, flow_matching, scripted) raise NotImplementedError until
their ``load`` / ``act`` methods are implemented.
"""

from policies.base import BasePolicy
from policies.ppo_multivariate import PPOMultivariatePolicy
from policies.ppo_normal import PPONormalPolicy
from policies.stubs import DiffusionPolicy, ScriptedWaypointPolicy
from policies.flow_matching import FlowMatchingPolicy

POLICY_REGISTRY: dict[str, type[BasePolicy]] = {
    "ppo_multivariate": PPOMultivariatePolicy,
    "ppo_normal": PPONormalPolicy,
    "diffusion": DiffusionPolicy,
    "flow_matching": FlowMatchingPolicy,
    "scripted": ScriptedWaypointPolicy,
}

__all__ = [
    "BasePolicy",
    "PPOMultivariatePolicy",
    "PPONormalPolicy",
    "DiffusionPolicy",
    "FlowMatchingPolicy",
    "ScriptedWaypointPolicy",
    "POLICY_REGISTRY",
]
