"""
KN Module - Shared components for K=N DDC experiments.

This module provides consistent interfaces between train and eval:
- config: Task/training/run configurations and constants
- prompt: Input/output formatting
- tokens: Token verification and initialization
- metrics: Evaluation metrics
- io: Data loading and run_config handling
"""

from .config import (
    DECISION_PREFIX,
    DECISION_ONLY,
    TOKENIZATION_POLICY,
    TaskConfig,
    TrainingConfig,
    K2_LOVE_CONFIG,
    K4_SUPPORT_CONFIG,
    TASK_CONFIGS,
)
from .prompt import format_input, format_output, log_prompt_format
from .tokens import (
    verify_single_token,
    get_mean_embedding,
    init_token_interpolated,
    log_embedding_geometry,
)
from .metrics import make_compute_metrics, probe_decision_priors
from .io import load_jsonl, load_run_config, RunConfig

__all__ = [
    # Constants
    "DECISION_PREFIX",
    "DECISION_ONLY",
    "TOKENIZATION_POLICY",
    # Configs
    "TaskConfig",
    "TrainingConfig",
    "K2_LOVE_CONFIG",
    "K4_SUPPORT_CONFIG",
    "TASK_CONFIGS",
    "RunConfig",
    # Prompt
    "format_input",
    "format_output",
    "log_prompt_format",
    # Tokens
    "verify_single_token",
    "get_mean_embedding",
    "init_token_interpolated",
    "log_embedding_geometry",
    # Metrics
    "make_compute_metrics",
    "probe_decision_priors",
    # IO
    "load_jsonl",
    "load_run_config",
]
