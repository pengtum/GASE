# 01 — The SEGA Framework

This folder documents the three required modules of the SEGA framework (paper §3.2):

| Module | Subfolder | Role |
|---|---|---|
| **Module 1** — Two-loop architecture with embedded geographic knowledge | `geoevolve_package/` | LLM-driven evolution engine. Adopted from the **GeoEvolve** framework (cited as `luo2025geoevolve` in the manuscript) and used via the `geoevolve` Python package. |
| **Module 2** — Automatic Code Formatter | `code_formatter/` | LLM-driven parser that converts a raw geospatial algorithm into the standardized triple (evolvable program, evaluator, configuration). |
| **Module 3** — Simulation-driven training protocol | `training_protocol/` | Specification of the train-on-simulation / apply-on-real protocol that keeps real-world results out-of-sample. |

Each subfolder has its own `README.md` with installation / usage notes.
