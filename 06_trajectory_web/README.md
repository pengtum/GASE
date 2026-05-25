# 06 — Trajectory Web (supplementary interactive view)

This is the interactive web page that accompanies **Figure 4** of the manuscript. It is hosted at:

> https://web-production-8126.up.railway.app/

The source below is the same code that runs at that URL. To run locally:

```bash
cd 06_trajectory_web
pip install -r requirements.txt
python build_data.py        # rebuilds the per-round artifacts from the source files
# then serve via gunicorn / flask per Procfile
```

---

# GeoRegime Web

Interactive visualization of the **GeoRegime** self-evolving spatial regime
regionalization algorithm — showing per-round metrics, RAG-retrieved
geographical knowledge, system-prompt edits, and the evolving Python
implementation across 10 rounds.

## What it shows

- **Evolution Timeline** — combined-score for each of the 10 rounds.
- **Auto-zoomed sparklines** for RandI, NMI, SSR, and MAE so even small
  per-round changes are visible.
- **RAG Knowledge Retrieval** — per round: missing/problematic knowledge,
  whether new geographical theory was needed, the keyword/category fetched,
  the search queries, and the full markdown-rendered retrieved knowledge.
- **Prompt Evolution** — the current system prompt and a unified diff against
  the previous round's `config_round_*.yaml`.
- **Code Evolution** — the best program at this round, in either Diff View
  (additions/deletions vs. the previous round) or Full Code with syntax
  highlighting.

## Layout

```
.
├── build_data.py        # parses raw outputs into web/data.json (see note below)
├── Procfile             # Railway start command
├── requirements.txt     # empty — tells Railway to provision Python
└── web/                 # static site (this is what gets deployed)
    ├── index.html
    ├── styles.css
    ├── app.js
    └── data.json        # pre-built; everything the page needs
```

The raw evolution outputs (`outputs_georegime_2kmodels_dynamic_gpt/`,
~12 MB of `config_round_*.yaml`, `round_*/best/best_program.py`, and the
master log) are **not** included in this repo — they're listed in
`.gitignore`. `web/data.json` is the rendered artifact and is everything
the page needs.

## Run locally

```bash
# If you have the raw outputs locally and want to rebuild data.json:
python build_data.py

# Serve the site:
python -m http.server 8000 --directory web
# then open http://localhost:8000/
```

## Deploy on GitHub Pages

1. Push this repo to GitHub.
2. **Settings → Pages → Source**: deploy from branch `main`, folder `/web`.
3. Visit `https://<your-user>.github.io/<repo>/`.

Alternatively, copy `web/` to a `gh-pages` branch and publish from root.

## Deploy on Railway

1. Push this repo to GitHub (or connect Railway directly to a local repo via
   the Railway CLI).
2. Create a new Railway project from the repo. Railway will detect the
   `Procfile` and start the static server.
3. Open the generated `*.up.railway.app` URL.

The included `Procfile` runs `python -m http.server $PORT --directory web` —
no extra dependencies, no build step. `.railwayignore` excludes the raw
`outputs_*/` data so the deployed image stays small.

## Rebuilding from new outputs

After producing new rounds in `outputs_georegime_2kmodels_dynamic_gpt/`, run:

```bash
python build_data.py
```

This regenerates `web/data.json` end-to-end.
