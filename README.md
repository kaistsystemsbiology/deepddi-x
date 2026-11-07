# DeepDDI-X

DeepDDI-X predicts drug–drug interaction types and severity from drug names and molecular structures. This repository contains the inference code.

## Installation

```bash
conda env create -f environment.yml
conda activate deepddi-x
```

## Data availability

DrugBank and Drugs.com data used in this study, including datasets derived from these sources, are not publicly distributed. Running the example requires the matching model checkpoints, configuration files, and reference and annotation files under `data/`.

## Example

Once the required files are available locally, run the following command from the repository root. Use a new or empty output directory.

```bash
python run_DeepDDI.py -i DDI_input.txt -o output
```

The final pair-level predictions are written to `output/unordered_pair_summary.tsv`.
