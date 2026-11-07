from __future__ import annotations
import hashlib
import json
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .mc_dropout import PROTOCOL, predict_shared
from .models import build_model
SEED = 20260901
MC_SAMPLES = 10
BATCH_SIZE = 64
THREADS = 4
ROOT = Path(__file__).resolve().parents[1]

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def setup_cpu() -> None:
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)

def load_model(path: str | Path, config_path: str | Path | None=None):
    path = Path(path)
    cfg = json.loads(Path(config_path or path.with_suffix('.json')).read_text())
    if sha256(path) != cfg['weights_sha256']:
        raise ValueError(f'Model checksum mismatch: {path.name}')
    if (cfg['protocol'], cfg['inference_seed'], cfg['mc_samples'], cfg['batch_size']) != (PROTOCOL, SEED, MC_SAMPLES, BATCH_SIZE):
        raise ValueError('The model configuration does not match the inference protocol')
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model({'model': cfg['model_config']}, int(cfg['input_dim']), int(cfg['output_dim']))
    model.load_state_dict(state, strict=True)
    model.eval()
    return (model, cfg)

def features_for_pairs(features: pd.DataFrame, pairs, representation: str) -> np.ndarray:
    a_ids, b_ids = zip(*pairs)
    missing = (set(a_ids) | set(b_ids)) - set(features.index)
    if missing:
        raise ValueError(f'Missing PCA50 IDs: {sorted(missing)[:20]}')
    a = features.loc[list(a_ids)].to_numpy(dtype=np.float32)
    b = features.loc[list(b_ids)].to_numpy(dtype=np.float32)
    if a.shape[1] != 50 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Each drug needs 50 finite PCA features')
    if representation == 'ordered_concat':
        return np.concatenate([a, b], axis=1)
    if representation == 'symmetric_sum_absdiff':
        return np.concatenate([a + b, np.abs(a - b)], axis=1)
    raise ValueError(f'Unsupported representation: {representation}')

def predict(model, bundle: dict, features: pd.DataFrame, pairs, task: str, progress=None):
    if not pairs:
        raise ValueError('Empty pair input')
    k = int(bundle['output_dim'])
    mean = np.empty((len(pairs), k), dtype=np.float32)
    sd = np.empty_like(mean)
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for start in range(0, len(pairs), 4096):
        sub = pairs[start:start + 4096]
        x = features_for_pairs(features, sub, bundle['feature_representation'])
        m, s, _ = predict_shared(model, bundle, x, sub, task=task, inference_seed=SEED, batch_size=BATCH_SIZE, mc_samples=MC_SAMPLES)
        mean[start:start + len(sub)] = m
        sd[start:start + len(sub)] = s
        if progress:
            progress(start + len(sub), len(pairs))
    if not all((torch.equal(v, state_before[key]) for key, v in model.state_dict().items())):
        raise AssertionError('Model tensors or BatchNorm statistics changed during inference')
    if not (np.isfinite(mean).all() and np.isfinite(sd).all() and (mean >= 0).all() and (mean <= 1).all() and (sd >= 0).all()):
        raise AssertionError('Invalid model probabilities')
    if task == 'severity' and (not np.allclose(mean.sum(1), 1, atol=1e-06)):
        raise AssertionError('Severity class probabilities do not sum to one')
    return (mean, sd)

def decisions(mean: np.ndarray, sd: np.ndarray, config: dict) -> dict:
    if config['protocol'] != PROTOCOL or config['inference_seed'] != SEED:
        raise ValueError('Inference configuration does not match this runtime')
    score = mean - float(config['uncertainty_weight']) * sd
    if config['task'] != 'severity':
        thresholds = np.asarray(config['label_thresholds'], dtype=np.float64)[None, :]
        return {'prediction': (score >= thresholds).astype(np.uint8), 'score': score}
    pred = mean.argmax(axis=1)
    confidence = score[np.arange(len(pred)), pred]
    accepted = confidence >= float(config['abstention_threshold'])
    return {'prediction': pred, 'score': score, 'confidence': confidence, 'accepted': accepted}

class Predictor:

    def __init__(self, task: str, root: str | Path=ROOT, checkpoint_dir=None):
        if task not in {'model1', 'model2', 'severity'}:
            raise ValueError('Unknown task')
        setup_cpu()
        self.root, self.task = (Path(root), task)
        directory = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else self.root / 'data/models'
        self.model, self.config = load_model(directory / f'{task}.pt')
        feature_path = self.root / 'data/drug_tanimoto_PCA50.csv'
        if sha256(feature_path) != self.config['feature_sha256']:
            raise ValueError('Frozen feature file checksum mismatch')
        self._lock = threading.RLock()

    def __call__(self, features: pd.DataFrame, pairs, progress=None) -> dict:
        pairs = list(pairs)
        with self._lock:
            mean, sd = predict(self.model, self.config, features, pairs, self.task, progress)
            return {'mean_probability': mean, 'std_probability': sd, **decisions(mean, sd, self.config)}
