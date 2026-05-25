# Module 2 — Automatic Code Formatter

This module turns a raw geospatial algorithm into the standardized triple required by the SEGA inner-loop engine:

```
Format(A_raw)  →  (A_init, E_eval, C_config)
```

- `A_init` — initial program with clearly marked `# EVOLVE-BLOCK-START / END` regions and a standardized `run_algorithm()` entry function
- `E_eval` — evaluator module that implements the evaluation protocol and returns a `combined_score`
- `C_config` — YAML configuration with all evolutionary hyperparameters and any task-specific settings

The formatter itself is implemented as part of the `geoevolve` package's automatic-formatter chain. A complete worked example on the Two-stage K-Models seed is given in **Appendix C** of the paper.

## How to use

```python
from geoevolve.formatter import Format
A_init, E_eval, C_config = Format(open("my_raw_algorithm.py").read())
```

## Worked example output

The worked example in the paper used `02_case_study_algorithms/seed/2kmodels_seed.py` as input, producing the following output files (see paper Appendix C, listings C.1–C.4):

- `initial_program.py` — formatter-produced `A_init`
- `evaluator.py` — formatter-produced `E_eval`
- `config.yaml` — formatter-produced `C_config`
- `_format_metadata.json` — provenance metadata (hashes, source paths)

(The exact run artifacts for the case study are inside the `geoevolve` package distribution.)
