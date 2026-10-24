"""Explicit numerical policy for paired GPU replay checks. AI-authored."""
import os

import torch


def configure_reproducible_execution():
    """Configure before CUDA initialization; unsupported kernels must fail.

    This enables deterministic PyTorch implementations where available. It
    cannot certify third-party kernels; the actual next-update replay remains
    required, with its original tolerances.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError('Configure deterministic execution before CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'actual_update_replay_still_required': True,
    }
