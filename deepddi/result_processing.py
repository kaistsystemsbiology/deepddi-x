"""Directional predictions and final unordered-pair reports for DeepDDI-X."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_unordered_pairs(
    directional: pd.DataFrame,
    results: dict[str, dict],
    configs: dict[str, dict],
) -> pd.DataFrame:
    """Combine both type-model directions within each prescription and pair.

    Severity is selected from the canonical order after checking that repeated
    records agree. If no type is retained in either order, only the final
    severity display becomes ``Not assessed``; the independent severity
    prediction and confidence remain available in separate columns.
    """
    work = directional.copy().reset_index(drop=True)
    work['_i'] = np.arange(len(work))
    work['unordered_pair_id'] = [
        '|'.join(sorted((str(a), str(b))))
        for a, b in zip(work.drug_a_id, work.drug_b_id)
    ]
    severity = results['severity']
    classes = configs['severity']['class_names']
    rows = []

    groups = work.groupby(
        ['Prescription', 'unordered_pair_id'], sort=False, dropna=False
    )
    for (prescription, pair_id), group in groups:
        indices = group['_i'].to_numpy(dtype=int)
        canonical = group[group.drug_a_id < group.drug_b_id]
        source_index = (
            int(canonical.iloc[0]['_i']) if len(canonical) else int(indices[0])
        )
        consistent = (
            np.all(
                severity['prediction'][indices]
                == severity['prediction'][source_index]
            )
            and np.all(
                severity['accepted'][indices]
                == severity['accepted'][source_index]
            )
            and np.allclose(
                severity['mean_probability'][indices],
                severity['mean_probability'][source_index],
                rtol=0,
                atol=1e-6,
            )
        )
        if not consistent:
            raise ValueError(
                f'Inconsistent repeated/order-invariant severity: {pair_id}'
            )

        drug_a, drug_b = pair_id.split('|')
        directions = set(zip(group.drug_a_id, group.drug_b_id))
        row = {
            'Prescription': prescription,
            'unordered_pair_id': pair_id,
            'drug_a_id': drug_a,
            'drug_b_id': drug_b,
            'n_directional_records': len(group),
            'n_evaluated_orders': len(directions),
            'both_orders_evaluated': int(
                (drug_a, drug_b) in directions and (drug_b, drug_a) in directions
            ),
            'source_sample_ids': json.dumps(group.sample_id.astype(str).tolist()),
            'severity_source_sample_id': str(work.iloc[source_index].sample_id),
        }
        retained_types = set()
        for task in ['model1', 'model2']:
            type_ids = sorted({
                int(value)
                for text in group[task + '_types']
                for value in str(text).split(';')
                if value
            })
            row[task + '_types'] = ';'.join(map(str, type_ids))
            retained_types.update(type_ids)

        raw_class = classes[int(severity['prediction'][source_index])].title()
        accepted = bool(severity['accepted'][source_index])
        raw_output = raw_class if accepted else 'Uncertain'
        row.update(
            ddi_type_ids=';'.join(map(str, sorted(retained_types))),
            n_ddi_types=len(retained_types),
            type_status='Type_retained' if retained_types else 'No_type_retained',
            severity_raw_class=raw_class,
            severity_raw_output=raw_output,
            severity_accepted_before_type_gating=int(accepted),
            severity=raw_output if retained_types else 'Not assessed',
            severity_confidence=float(severity['confidence'][source_index]),
            severity_threshold=float(configs['severity']['abstention_threshold']),
        )
        for j, class_name in enumerate(classes):
            row[class_name + '_probability'] = float(
                severity['mean_probability'][source_index, j]
            )
            row[class_name + '_std'] = float(
                severity['std_probability'][source_index, j]
            )
        rows.append(row)

    return pd.DataFrame(rows)


def write_results(
    samples: pd.DataFrame,
    results: dict[str, dict],
    configs: dict[str, dict],
    output_dir: str | Path,
    root: str | Path,
) -> None:
    """Write directional details, the final pair summary, and raw probabilities.

    ``unordered_pair_summary.tsv`` is the final pair-level output. The original
    directional tables remain available for detailed inspection and existing
    downstream readers.
    """
    out, root = Path(output_dir), Path(root)
    sev = results['severity']
    classes = configs['severity']['class_names']
    severity = samples.copy()
    severity['predicted_class'] = [
        classes[int(value)].title() for value in sev['prediction']
    ]
    severity['accepted'] = sev['accepted'].astype(int)
    severity['severity'] = [
        classes[int(value)].title() if accepted else 'Uncertain'
        for value, accepted in zip(sev['prediction'], sev['accepted'])
    ]
    severity['confidence_score'] = sev['confidence']
    for j, name in enumerate(classes):
        severity[f'{name}_probability'] = sev['mean_probability'][:, j]
        severity[f'{name}_std'] = sev['std_probability'][:, j]
    severity.to_csv(out / 'severity_result.txt', sep='\t', index=False)

    final_rows = []
    summaries = samples.copy()
    for task, known_name in [
        ('model1', 'DrugBank_known_ddi.txt'),
        ('model2', 'Model2_known_ddi.txt'),
    ]:
        config, result = configs[task], results[task]
        folder = out / f'{task}_DDI_results'
        folder.mkdir()
        templates = pd.read_csv(
            root / f'data/Type_information/Interaction_information_{task}.csv',
            sep='\t',
            dtype={'type': int},
        )
        sentences = dict(zip(templates['type'], templates['sentence']))
        known = pd.read_csv(root / 'data' / known_name, sep='\t')
        recorded = {
            (str(a), str(b), int(type_id))
            for a, b, type_id in known.itertuples(index=False, name=None)
        }
        rows = []
        selected_text = []
        for i, sample in enumerate(samples.to_dict('records')):
            selected = np.flatnonzero(result['prediction'][i])
            selected_text.append(
                ';'.join(str(config['label_ids'][j]) for j in selected)
            )
            for j in selected:
                type_id = int(config['label_ids'][j])
                sentence = (
                    str(sentences[type_id])
                    .replace('#Drug1', sample['drug_a_name'])
                    .replace('#Drug2', sample['drug_b_name'])
                )
                row = {
                    **sample,
                    'model': task,
                    'type': type_id,
                    'sentence': sentence,
                    'mean_probability': float(result['mean_probability'][i, j]),
                    'probability_std': float(result['std_probability'][i, j]),
                    'decision_score': float(result['score'][i, j]),
                    'threshold': float(config['label_thresholds'][j]),
                    'recorded_reference': int(
                        (sample['drug_a_id'], sample['drug_b_id'], type_id)
                        in recorded
                    ),
                    'severity': severity.iloc[i]['severity'],
                    'severity_accepted': int(sev['accepted'][i]),
                }
                rows.append(row)
                final_rows.append(row)

        columns = list(samples.columns) + [
            'model', 'type', 'sentence', 'mean_probability', 'probability_std',
            'decision_score', 'threshold', 'recorded_reference', 'severity',
            'severity_accepted',
        ]
        table = pd.DataFrame(rows, columns=columns)
        table.drop(
            columns=['sentence', 'recorded_reference', 'severity', 'severity_accepted']
        ).to_csv(folder / 'DDI_result.txt', sep='\t', index=False)
        table.drop(columns=['recorded_reference']).to_csv(
            folder / 'Final_DDI_result.txt', sep='\t', index=False
        )
        table.to_csv(folder / 'Final_annotated_DDI_result.txt', sep='\t', index=False)
        summaries[f'{task}_types'] = selected_text

    summaries['severity'] = severity['severity'].to_numpy()
    summaries['severity_accepted'] = sev['accepted'].astype(int)
    summaries['type_status'] = np.where(
        (summaries.model1_types != '') | (summaries.model2_types != ''),
        'Type_retained',
        'No_type_retained',
    )
    summaries.to_csv(out / 'pair_summary.tsv', sep='\t', index=False)
    summaries.to_csv(out / 'ordered_pair_summary.tsv', sep='\t', index=False)
    summarize_unordered_pairs(summaries, results, configs).to_csv(
        out / 'unordered_pair_summary.tsv', sep='\t', index=False
    )
    pd.DataFrame(final_rows, columns=columns).to_csv(
        out / 'Final_result.txt', sep='\t', index=False
    )
    for task, result in results.items():
        np.savez_compressed(
            out / f'{task}_probabilities.npz',
            sample_ids=samples.sample_id.to_numpy(dtype=str),
            drug_a_ids=samples.drug_a_id.to_numpy(dtype=str),
            drug_b_ids=samples.drug_b_id.to_numpy(dtype=str),
            **result,
        )
