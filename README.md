# DeepDDI-X

## Project
DeepDDI-X predicts drug-drug interaction (DDI) types and severity from drug names and molecular structures. Model 1 covers 113 DrugBank-derived types; Model 2 covers 88 additional Drugs.com-derived types. A separate ordinal model predicts Minor, Moderate, or Major, with Uncertain for predictions below the saved confidence threshold.

## Package structure

```text
DeepDDIX/
├── run_DeepDDI.py       # Command-line entry point
├── deepddi/            # Input preparation, prediction, and result processing
├── data/               # Model weights, PCA transform, and reference data
├── DDI_input.txt       # Example input
├── environment.yml
├── README.md
└── LICENSE
```

`deepddi/preprocessing.py` resolves drug identities, prepares structure features, and completes missing reverse input records. `deepddi/result_processing.py` writes directional predictions and the final unordered-pair summary. Both are called by `run_DeepDDI.py`; no separate postprocessing command is required. Model weights and saved inference settings are included under `data/models/`.

Training, benchmark experiments, and manuscript analyses are maintained in the separate reproduction materials.

## Installation and example
The supplied environment targets Linux and Python 3.11. Run from this directory:

```bash
conda env create -f environment.yml
conda activate deepddi-x
python run_DeepDDI.py -i ./DDI_input.txt -o ./output
```

Use a new or empty output directory. For a second run, choose another name, such as `./output_2`. Results are written only to the specified output directory.

## Input
`DDI_input.txt` is a tab-separated file with `Prescription`, `Drug name`, and `Smiles` columns. Drugs with the same prescription identifier are paired in both orders. Recognized names/IDs can use a blank SMILES; a supplied structure must match the recognized identity. An unrecognized drug requires a valid SMILES. Duplicate identities within one prescription are rejected. An optional `DrugBank ID` column resolves ambiguous names to a registered ID. The identity table includes the structural reference set and application drugs from Dataset S6. Registered drugs retain the same DrugBank ID whether their features are frozen or projected from structure. For a registered nonreference drug, a blank SMILES uses its bundled application structure. Supplied structures must match the identity table; ambiguous structure-to-ID matches require an explicit ID. Only unregistered structures receive a `SMILES:` hash ID.

A pair table with `drug_a_id` and `drug_b_id` columns is also accepted for registered IDs; `sample_id` and `Prescription` are optional. Missing reverse pairs are added automatically within each prescription. Different prescriptions remain separate.

New structures use the supplied reference structures and PCA transform. Before projecting a new structure, the runtime checks reconstruction of all 2,386 reference feature rows and stops on a mismatch. This numerical check is **not** evidence of predictive validity for unseen drugs. Optional `-p` accepts only a PCA50 CSV whose requested rows match the validated model features.

## Output
**Read `output/unordered_pair_summary.tsv` for final pair-level results.** Each prescription and unordered pair has one row. Retained types are the union of Model 1 and Model 2 predictions across both orders. The `severity` field is:

- **Minor / Moderate / Major:** a type was retained and severity passed the saved threshold.
- **Uncertain:** a type was retained, but severity did not pass the threshold.
- **Not assessed:** neither type model retained any type in either order.

The final table retains the raw severity class, probabilities, confidence score, and pre-gating output for traceability. `Not assessed` overrides the final display only; it does not erase the independent severity prediction.

`Final_result.txt` and the two `model*_DDI_results/` directories retain **directional** type statements. `severity_result.txt` is the independent, pre-type-gating severity output. For compatibility, `pair_summary.tsv` remains a directional table; `ordered_pair_summary.tsv` is its explicitly named copy. Neither is the final unordered summary. Probability archives, resolved inputs, and `prediction_manifest.json` are written to the chosen output directory. The manifest records the identity-table checksum. MC dropout uses the resolved drug IDs, model task, inference seed, draw, and layer as its key, so reproducing a published prediction requires the same resolved IDs as well as the same features and model settings.

## Interpretation
The type models were not trained as interaction-present/absent classifiers. No retained type, or `Not assessed`, does not establish that a combination is safe. Uncertain is an abstention status, not a fourth trained severity class. The saved severity cutoff is 0.64 on the selected class's mean probability minus 0.5 times its MC standard deviation; it is not a guarantee of 64% correctness. A `recorded_reference` flag denotes a match in the bundled source table, not independent clinical validation.

This is research software, not a substitute for clinical assessment. Generalization beyond the training drugs needs separate evaluation. See the retained `LICENSE` and third-party notice.
