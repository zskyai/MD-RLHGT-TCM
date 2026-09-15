# MD-RLHGT-TCM

Reproducible implementation and released artifacts for **MD-RLHGT**, a topology-adaptive heterogeneous graph model for composition-informed herb--target prioritization in traditional Chinese medicine.

## Contents

- Python scripts in the repository root provide data processing, model definition, training, inference, evaluation, ablation and comparison workflows.
- `processed/`: the processed heterogeneous graph, feature tensors, entity index and herb--compound--target pair manifest used by the released checkpoint.
- `checkpoints/best_model.pt`: the released model checkpoint.

The large PyTorch artifacts are tracked with Git LFS. The released processed files are the exact artifacts consumed by the checkpoint; they avoid any dependence on undocumented preprocessing state.

## Environment

The experiments were run with Python 3.10+, PyTorch, PyTorch Geometric and the scientific Python stack. Install the versions appropriate for the target CUDA runtime, then install the project dependencies used by the scripts. GPU selection is exposed by the command-line arguments of the training and evaluation scripts where applicable.

## Reproduce the released analyses

Run commands from the repository root. The processed graph and checkpoint are already included:

```bash
python evaluate.py --help
python unified_cold_start.py --help
python ablation.py --help
```

For a fresh preprocessing run, inspect `processed.py` and `generate_data.py` and provide the source tables expected by those scripts. The released processed artifacts should be used when reproducing the manuscript tables and rankings.

## Data and model scope

The repository contains processed, release-ready artifacts and the model checkpoint used in the manuscript. It does not include private credentials, caches, logs, temporary experiment folders, local pretrained-model caches or unrelated exploratory outputs. Source tables should be redistributed only when their licensing and provenance permit it.

## Citation

Please cite the accompanying Bioinformatics manuscript when using this code or the released artifacts. The manuscript provides the evaluation protocol, data split definition, model description and evidence assessment.

## License

The repository is released for research use. Dataset and third-party model terms remain applicable to the corresponding upstream resources.

