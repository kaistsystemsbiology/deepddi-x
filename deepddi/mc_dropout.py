from __future__ import annotations
import hashlib
import json
from typing import Sequence
import numpy as np
import torch
from torch import nn
PROTOCOL = 'deepddix-pair-keyed-mc-v1'
TASKS = {'model1', 'model2', 'severity'}

def keep_mask(pairs: Sequence[tuple[str, str]], *, task: str, draw: int, layer: int, width: int, dropout_p: float, inference_seed: int=20260901) -> np.ndarray:
    if task not in TASKS:
        raise ValueError(f'task must be one of {sorted(TASKS)}')
    if min(draw, layer, inference_seed) < 0 or width < 1:
        raise ValueError('Nonnegative indices/seed and positive width required')
    if not np.isfinite(dropout_p) or not 0 <= dropout_p < 1:
        raise ValueError('dropout_p must be in [0, 1)')
    out = np.empty((len(pairs), width), dtype=np.float32)
    for i, pair in enumerate(pairs):
        if len(pair) != 2 or not all((isinstance(x, str) and x for x in pair)):
            raise ValueError('Each pair must contain two nonempty stable drug IDs')
        drug_ids = sorted(pair) if task == 'severity' else list(pair)
        key = [PROTOCOL, int(inference_seed), task, drug_ids, int(draw), int(layer)]
        raw = json.dumps(key, ensure_ascii=True, separators=(',', ':')).encode('ascii')
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:16], 'little')
        rng = np.random.Generator(np.random.PCG64(seed))
        out[i] = (rng.random(width) >= dropout_p).astype(np.float32)
    return out

def forward_with_masks(model: nn.Module, x: torch.Tensor, masks: Sequence[np.ndarray]) -> torch.Tensor:
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    if len(drops) != len(masks):
        raise ValueError('Expected one mask per nn.Dropout layer')
    if any((isinstance(m, nn.modules.batchnorm._BatchNorm) and (not m.track_running_stats) for m in model.modules())):
        raise ValueError('BatchNorm must use stored running statistics')
    modes = [(m, m.training) for m in model.modules()]
    hooks = []
    calls = [0] * len(drops)
    try:
        model.eval()
        for i, (drop, mask) in enumerate(zip(drops, masks)):
            arr = np.asarray(mask, dtype=np.float32)
            if not np.isin(arr, [0.0, 1.0]).all():
                raise ValueError('Masks must be unscaled 0/1 arrays')
            if not 0 <= drop.p < 1:
                raise ValueError('Unsupported dropout probability')

            def apply(mod, inputs, output, *, idx=i, saved=arr):
                del output
                v = inputs[0]
                if tuple(v.shape) != saved.shape:
                    raise ValueError('Dropout activation and mask shapes differ')
                calls[idx] += 1
                mask_t = torch.as_tensor(saved, dtype=v.dtype, device=v.device)
                return v * mask_t / (1.0 - mod.p)
            hooks.append(drop.register_forward_hook(apply))
        with torch.inference_mode():
            result = model(x)
        if calls != [1] * len(drops):
            raise ValueError('Only sequential MLPs using each dropout once are supported')
        return result
    finally:
        for hook in hooks:
            hook.remove()
        for module, previous_mode in modes:
            module.training = previous_mode

def probabilities(logits: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == 'multilabel_mlp':
        return torch.sigmoid(logits)
    if kind == 'ordinal_mlp':
        q = torch.sigmoid(logits)
        p = torch.cat([1 - q[:, :1], q[:, :-1] - q[:, 1:], q[:, -1:]], dim=1)
        p = torch.clamp(p, min=0)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    raise ValueError(f'Unsupported model kind: {kind}')

def predict_shared(model: nn.Module, bundle: dict, X: np.ndarray, pairs: Sequence[tuple[str, str]], *, task: str, inference_seed: int=20260901, batch_size: int=64, mc_samples: int | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or len(X) != len(pairs) or len(X) == 0 or (not np.isfinite(X).all()):
        raise ValueError('Expected a finite nonempty [N,D] feature array and N pairs')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    cfg = bundle['model_config']
    dims = [int(n) for n in cfg['hidden_dims']]
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    if len(dims) != len(drops):
        raise ValueError('Hidden dimension/dropout count mismatch')
    draws_n = int(mc_samples if mc_samples is not None else (bundle.get('uncertainty') or {}).get('mc_samples', 10))
    if draws_n < 2:
        raise ValueError('MC inference requires at least two different draws')
    temperature = float(bundle.get('temperature', 1.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid saved temperature')
    device = next(model.parameters()).device
    results = []
    for draw in range(draws_n):
        batches = []
        for start in range(0, len(X), batch_size):
            subpairs = pairs[start:start + batch_size]
            masks = [keep_mask(subpairs, task=task, draw=draw, layer=i, width=width, dropout_p=float(drop.p), inference_seed=inference_seed) for i, (width, drop) in enumerate(zip(dims, drops))]
            x = torch.as_tensor(X[start:start + batch_size], device=device)
            logits = forward_with_masks(model, x, masks) / temperature
            batches.append(probabilities(logits, cfg['kind']).cpu().numpy())
        results.append(np.concatenate(batches, axis=0))
    draws = np.stack(results, axis=0)
    return (draws.mean(axis=0), draws.std(axis=0, ddof=0), draws)
