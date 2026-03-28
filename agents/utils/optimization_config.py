#!/usr/bin/env python3
"""Shared optimization config model and parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_OPTIMIZATION_FREQUENCY = 10


@dataclass(frozen=True)
class OptimizationConfig:
    """Canonical optimization configuration for PokeAgent and HarnessEvolver."""

    enable_prompt_evolve: bool = False
    enable_subagent_evolve: bool = False
    enable_skill_evolve: bool = False
    enable_memory_evolve: bool = False
    optimization_frequency: int = DEFAULT_OPTIMIZATION_FREQUENCY
    trajectory_window_steps: Optional[int] = None

    def any_enabled(self) -> bool:
        return any(
            (
                self.enable_prompt_evolve,
                self.enable_subagent_evolve,
                self.enable_skill_evolve,
                self.enable_memory_evolve,
            )
        )

    def any_harness_pass_enabled(self) -> bool:
        return any(
            (
                self.enable_prompt_evolve,
                self.enable_subagent_evolve,
                self.enable_skill_evolve,
                self.enable_memory_evolve,
            )
        )

    def has_non_prompt_passes(self) -> bool:
        return any(
            (
                self.enable_subagent_evolve,
                self.enable_skill_evolve,
                self.enable_memory_evolve,
            )
        )

    def with_non_prompt_passes_disabled(self) -> "OptimizationConfig":
        return replace(
            self,
            enable_subagent_evolve=False,
            enable_skill_evolve=False,
            enable_memory_evolve=False,
        )

    def resolved_trajectory_window_steps(self) -> int:
        return self.trajectory_window_steps or self.optimization_frequency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_prompt_evolve": self.enable_prompt_evolve,
            "enable_subagent_evolve": self.enable_subagent_evolve,
            "enable_skill_evolve": self.enable_skill_evolve,
            "enable_memory_evolve": self.enable_memory_evolve,
            "optimization_frequency": self.optimization_frequency,
            "trajectory_window_steps": self.trajectory_window_steps,
        }


def scaffold_optimization_defaults(scaffold: str) -> OptimizationConfig:
    scaffold_name = (scaffold or "").strip().lower()
    if scaffold_name == "autoevolve":
        return OptimizationConfig(
            enable_prompt_evolve=True,
            enable_subagent_evolve=True,
            enable_skill_evolve=True,
            enable_memory_evolve=True,
        )
    if scaffold_name in {"pokeagent", "autonomous_cli"}:
        return OptimizationConfig(enable_prompt_evolve=True)
    return OptimizationConfig()


def parse_optimization_config_string(raw_json: str) -> Dict[str, Any]:
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("--optimization-config JSON must decode to an object")
    return parsed


def load_optimization_config_file(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if not isinstance(parsed, dict):
        raise ValueError("--optimization-config-file must contain a JSON object")
    return parsed


def _apply_dict_values(base: OptimizationConfig, config_data: Dict[str, Any]) -> OptimizationConfig:
    cfg = base
    for key in (
        "enable_prompt_evolve",
        "enable_subagent_evolve",
        "enable_skill_evolve",
        "enable_memory_evolve",
    ):
        if key in config_data and config_data[key] is not None:
            cfg = replace(cfg, **{key: bool(config_data[key])})
    if "optimization_frequency" in config_data and config_data["optimization_frequency"] is not None:
        cfg = replace(cfg, optimization_frequency=max(1, int(config_data["optimization_frequency"])))
    if "trajectory_window_steps" in config_data:
        value = config_data["trajectory_window_steps"]
        cfg = replace(cfg, trajectory_window_steps=None if value is None else max(1, int(value)))
    return cfg


def build_optimization_config(
    *,
    scaffold: str,
    config_data: Optional[Dict[str, Any]] = None,
    convenience_overrides: Optional[Dict[str, Any]] = None,
    legacy_enable_prompt_optimization: Optional[bool] = None,
    legacy_optimization_frequency: Optional[int] = None,
) -> OptimizationConfig:
    """
    Build effective optimization config from scaffold defaults + overrides.

    Resolution order:
    1) Disabled baseline unless a config source is provided.
    2) Deprecated legacy flags (if present).
    3) Structured config object values.
    4) Convenience override flags.
    """

    has_structured_input = bool(config_data) or bool(convenience_overrides)
    if has_structured_input:
        cfg = scaffold_optimization_defaults(scaffold)
    else:
        cfg = OptimizationConfig()

    if legacy_enable_prompt_optimization is True:
        cfg = scaffold_optimization_defaults(scaffold)
    elif legacy_enable_prompt_optimization is False:
        cfg = OptimizationConfig()

    if legacy_optimization_frequency is not None:
        cfg = replace(cfg, optimization_frequency=max(1, int(legacy_optimization_frequency)))

    if config_data:
        cfg = _apply_dict_values(cfg, config_data)
    if convenience_overrides:
        cfg = _apply_dict_values(cfg, convenience_overrides)

    return cfg
