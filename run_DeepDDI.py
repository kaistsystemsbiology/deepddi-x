"""Command-line inference with complete directional types and final pair summaries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deepddi.DeepDDI import ROOT, Predictor, sha256
from deepddi.Severity import SeverityPredictor
from deepddi.preprocessing import FEATURE_PROTOCOL, IDENTITY_PROTOCOL, complete_directions, prepare_input
from deepddi.result_processing import write_results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Predict directional DDI types and ordinal severity; '
                    'write one final summary row per unordered pair and prescription.')
    parser.add_argument('-i', '--input_file', required=True,
                        help='Prescription/Drug name/Smiles TSV, or drug_a_id/drug_b_id TSV')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='A new or empty output directory')
    parser.add_argument('-p', '--PCA_profile_file',
                        help='Optional PCA50 CSV indexed by stable drug ID; '
                             'must match the validated model features')
    args = parser.parse_args(argv)
    out = Path(args.output_dir).expanduser().resolve()
    protected = [ROOT / 'data', ROOT / 'deepddi']
    if out == ROOT or any(out == p or p in out.parents for p in protected):
        parser.error('Do not write outputs into release assets')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Output directory must be new or empty; choose a different directory')

    samples, features, mapping = prepare_input(
        args.input_file, ROOT, args.PCA_profile_file)
    n_original_records = len(samples)
    # Type models are directional; both orders are needed for the final union.
    samples = complete_directions(samples)
    out.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out / 'input_mapping.tsv', sep='\t', index=False)
    samples.to_csv(out / 'DDI_input.txt', sep='\t', index=False)
    used = sorted(set(samples.drug_a_id) | set(samples.drug_b_id))
    features.loc[used].to_csv(out / 'PCA_transformed_similarity_profile.csv')

    results, configs = {}, {}
    pairs = list(zip(samples.drug_a_id, samples.drug_b_id))
    for task in ['model1', 'model2', 'severity']:
        engine = SeverityPredictor(ROOT) if task == 'severity' else Predictor(task, ROOT)
        print(f'{task}: {len(samples)} ordered input records', flush=True)
        results[task] = engine(
            features, pairs,
            lambda n, k: print(f'{task} {n}/{k}', flush=True))
        configs[task] = engine.config

    write_results(samples, results, configs, out, ROOT)
    run = {
        'software': 'DeepDDI-X',
        'release_layout': 'compact-inference',
        'n_original_records': n_original_records,
        'n_ordered_records': len(samples),
        'complete_directions': True,
        'primary_summary': 'unordered_pair_summary.tsv',
        'legacy_summary': 'pair_summary.tsv',
        'input_sha256': sha256(args.input_file),
        'feature_protocol': FEATURE_PROTOCOL,
        'identity_protocol': IDENTITY_PROTOCOL,
        'identity_table_sha256': sha256(ROOT / 'data/drug_identity.tsv'),
        'feature_protocol_sha256': sha256(ROOT / 'data/feature_protocol.json'),
        'feature_table_sha256': sha256(ROOT / 'data/drug_tanimoto_PCA50.csv'),
        'inference_protocol': configs['model1']['protocol'],
        'inference_seed': configs['model1']['inference_seed'],
        'mc_samples': configs['model1']['mc_samples'],
        'models': {
            task: {
                'weights_sha256': config['weights_sha256'],
                'configuration_sha256': sha256(ROOT / f'data/models/{task}.json'),
            }
            for task, config in configs.items()
        },
    }
    (out / 'prediction_manifest.json').write_text(json.dumps(run, indent=2) + '\n')
    print(f'OUTPUT={out}\nPRIMARY_SUMMARY={out / "unordered_pair_summary.tsv"}'
          '\nPREDICTION_COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
