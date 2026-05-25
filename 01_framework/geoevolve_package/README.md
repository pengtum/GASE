# Module 1 — Two-Loop Architecture (via the `geoevolve` package)

This module is **not redistributed here** — we adopt the open-source `geoevolve` Python package distributed alongside Luo et al. (2025), which already implements the complete two-loop architecture (Code Evolver, Code Analyzer, Knowledge Retriever / GeoKnowRAG, Geo-informed Prompt Generator).

## Installation

```bash
pip install geoevolve
```

(or install from source — see the GeoEvolve repository linked in the paper).

## Usage in this release

Both the simulation and real-world runners in `03_*/` and `04_*/` call into `geoevolve` for any "GeoEvolve_*" variant. The configuration is identical to the original GeoEvolve paper except for the case-study-specific settings (seed algorithm, simulator, fitness signal, target number of regions, spatial weights) documented in the paper's Appendix B.

The exact configuration used in this paper:
- LLM backbone: `openai/gpt-4o` (primary, weight 0.8) + `openai/gpt-4.1` (secondary, weight 0.2), via OpenRouter
- Sampling: temperature 0.7, top_p 0.95, max_tokens 8192
- Search: MAP-Elites with population 50, archive 20, 5 islands; selection ratios (0.1, 0.2, 0.7); 10×10 (score, complexity) feature map
- Budget: 100 inner-loop iterations per variant; for GeoEvolve variants this is 10 outer-loop cycles × 10 inner iterations
- Each evaluation: 100 s timeout, up to 3 retries
- GeoKnowRAG corpus: 141 documents, embedded with `text-embedding-3-small`, stored in Chroma, retrieved with RAG-Fusion (reciprocal rank fusion)

## Reference

> Luo, P. et al. (2025). GeoEvolve: An LLM-Powered Multi-Agent System for Evolving Geospatial Algorithms with Embedded Geographic Knowledge.

The PDF is included in `docs/geoevolve_paper_2025.pdf`.
