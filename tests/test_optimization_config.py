#!/usr/bin/env python3
"""Seam tests for OptimizationConfig resolution and HarnessEvolver pass gating."""

from types import SimpleNamespace

from agents.utils.harness_evolver import HarnessEvolver
from agents.utils.optimization_config import (
    OptimizationConfig,
    build_optimization_config,
    parse_optimization_config_string,
    scaffold_optimization_defaults,
)


def test_scaffold_defaults_match_section11_intent():
    pokeagent_cfg = scaffold_optimization_defaults("pokeagent")
    autoevolve_cfg = scaffold_optimization_defaults("autoevolve")
    simple_cfg = scaffold_optimization_defaults("simple")

    assert pokeagent_cfg.enable_prompt_evolve is True
    assert pokeagent_cfg.enable_subagent_evolve is False
    assert pokeagent_cfg.enable_skill_evolve is False
    assert pokeagent_cfg.enable_memory_evolve is False

    assert autoevolve_cfg.enable_prompt_evolve is True
    assert autoevolve_cfg.enable_subagent_evolve is True
    assert autoevolve_cfg.enable_skill_evolve is True
    assert autoevolve_cfg.enable_memory_evolve is True

    assert simple_cfg.any_enabled() is False


def test_build_optimization_config_legacy_and_overrides():
    cfg = build_optimization_config(
        scaffold="autoevolve",
        legacy_enable_prompt_optimization=True,
        legacy_optimization_frequency=25,
        convenience_overrides={"enable_skill_evolve": False},
    )
    assert cfg.enable_prompt_evolve is True
    assert cfg.enable_subagent_evolve is True
    assert cfg.enable_skill_evolve is False
    assert cfg.enable_memory_evolve is True
    assert cfg.optimization_frequency == 25


def test_build_optimization_config_structured_values():
    cfg = build_optimization_config(
        scaffold="pokeagent",
        config_data={
            "enable_subagent_evolve": True,
            "optimization_frequency": 40,
            "trajectory_window_steps": 60,
        },
    )
    assert cfg.enable_prompt_evolve is True
    assert cfg.enable_subagent_evolve is True
    assert cfg.optimization_frequency == 40
    assert cfg.resolved_trajectory_window_steps() == 60


def test_parse_optimization_config_requires_json_object():
    parsed = parse_optimization_config_string('{"enable_prompt_evolve": true}')
    assert parsed["enable_prompt_evolve"] is True


def _make_harness_for_evolve(cfg: OptimizationConfig):
    harness = object.__new__(HarnessEvolver)
    harness.optimization_config = cfg
    harness.prompt_optimizer = SimpleNamespace(get_recent_trajectories=lambda _n: [{"step": 1}])
    harness.generation = 0
    harness._prev_skill_stats = {}
    harness._prev_changes_summary = ""
    harness._save_evolution_log = lambda *_args, **_kwargs: None
    return harness


def test_harness_evolve_skips_disabled_passes():
    cfg = OptimizationConfig(
        enable_prompt_evolve=True,
        enable_subagent_evolve=False,
        enable_skill_evolve=False,
        enable_memory_evolve=False,
    )
    harness = _make_harness_for_evolve(cfg)

    calls = {"prompt": 0, "subagents": 0, "skills": 0, "memory": 0}
    harness._compute_skill_stats = lambda _t: {}
    harness._auto_revert_degraded_skills = lambda _stats: []

    def _prompt(_step, _window):
        calls["prompt"] += 1
        return {"updated": True}

    def _subagents(_traj, _step):
        calls["subagents"] += 1
        return {"created": 1}

    def _skills(_traj, _step):
        calls["skills"] += 1
        return {"updated": 1}

    def _memory(_traj, _step):
        calls["memory"] += 1
        return {"updated": 1}

    harness._evolve_prompt = _prompt
    harness._evolve_subagents = _subagents
    harness._evolve_skills = _skills
    harness._evolve_memory = _memory

    out = harness.evolve(current_step=100, num_trajectory_steps=25)
    assert out["prompt"]["updated"] is True
    assert out["subagents"]["skipped"] is True
    assert out["skills"]["skipped"] is True
    assert out["memory"]["skipped"] is True
    assert calls == {"prompt": 1, "subagents": 0, "skills": 0, "memory": 0}


def test_harness_evolve_all_passes_disabled():
    harness = _make_harness_for_evolve(OptimizationConfig())
    out = harness.evolve(current_step=100, num_trajectory_steps=25)
    assert out == {"skipped": True, "reason": "all_passes_disabled"}
