"""Parse GeoRegime evolution outputs into a single JSON file for the web visualization.

Reads:
  - outputs_georegime_2kmodels_dynamic_gpt/round_<N>/best/best_program.py and best_program_info.json
  - outputs_georegime_2kmodels_dynamic_gpt/config_round_<N>.yaml
  - outputs_georegime_2kmodels_dynamic_gpt/geoevolve_georegime_2kmodels_dynamic_gpt.log

Writes:
  - web/data.json
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs_georegime_2kmodels_dynamic_gpt"
WEB_DIR = ROOT / "web"
WEB_DIR.mkdir(exist_ok=True)

NUM_ROUNDS = 10
LOG_PATH = DATA_DIR / "geoevolve_georegime_2kmodels_dynamic_gpt.log"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict:
    return json.loads(read_text(path))


def parse_log_sections(log_text: str) -> list[dict]:
    """Split the log into per-round sections.

    Each section starts with 'Round N' (or after a separator line containing
    100 '=' followed immediately by 'Round N'). We keep only the LAST run, which
    is the one that produced the round_1..round_10 outputs (rounds 0..8 in the
    log map to round_1..round_9 in the folders; round_10 has no log entry).
    """
    # Find all positions where a section header appears.
    # Header pattern: ====...Round N (or beginning of file: 'Round 0')
    header_re = re.compile(r"(?m)^=+Round\s+(\d+)\s*$|^Round\s+(\d+)\s*$")
    headers = list(header_re.finditer(log_text))

    sections: list[tuple[int, int, int]] = []  # (round_num, start, end)
    for i, m in enumerate(headers):
        rn = int(m.group(1) or m.group(2))
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(log_text)
        sections.append((rn, start, end))

    # Group sections by 'run': a run is a sequence of round numbers that resets
    # back to 0. Use the LAST run.
    runs: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    last_rn = -1
    for sec in sections:
        rn = sec[0]
        if rn == 0 and current:
            runs.append(current)
            current = []
        current.append(sec)
        last_rn = rn
    if current:
        runs.append(current)

    final_run = runs[-1] if runs else []

    parsed: list[dict] = []
    for rn, start, end in final_run:
        body = log_text[start:end]
        parsed.append({"round_num": rn, "body": body})
    return parsed


def parse_round_body(body: str) -> dict:
    """Extract metric, knowledge needed, knowledge retrieved, prompt updated."""
    out = {
        "metric": None,
        "knowledge_needed": None,
        "knowledge_retrieved": "",
        "prompt_updated": "",
    }

    # metric line
    m = re.search(r"^metric:\s*(\{.*\})\s*$", body, re.MULTILINE)
    if m:
        try:
            out["metric"] = ast.literal_eval(m.group(1))
        except Exception:
            out["metric"] = None

    # knowledge needed: a python-dict literal on the line after the label
    kn_idx = body.find("knowledge needed:")
    kr_idx = body.find("knowledge retrieved:")
    pu_idx = body.find("prompt updated:")

    if kn_idx >= 0 and kr_idx > kn_idx:
        kn_block = body[kn_idx + len("knowledge needed:"):kr_idx].strip()
        # The dict is usually on the first non-empty line; but it can wrap
        try:
            out["knowledge_needed"] = ast.literal_eval(kn_block)
        except Exception:
            # Try just the first line
            first_line = kn_block.splitlines()[0].strip() if kn_block else ""
            try:
                out["knowledge_needed"] = ast.literal_eval(first_line)
            except Exception:
                out["knowledge_needed"] = {"_raw": kn_block[:1000]}

    if kr_idx >= 0:
        end_idx = pu_idx if pu_idx > kr_idx else len(body)
        out["knowledge_retrieved"] = body[kr_idx + len("knowledge retrieved:"):end_idx].strip()

    if pu_idx >= 0:
        out["prompt_updated"] = body[pu_idx + len("prompt updated:"):].strip()

    return out


def extract_yaml_system_message(yaml_text: str) -> str:
    """Cheap extractor for the prompt.system_message block in our YAML.

    Avoids requiring PyYAML. Looks for `system_message: |` (or `|-`) and reads
    the indented block until a less-indented line (e.g. evaluator_system_message).
    """
    m = re.search(r"^\s*system_message:\s*\|[-+]?\s*\n", yaml_text, re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    # Determine indent of the block by reading the first non-empty line
    rest = yaml_text[start:]
    lines = rest.splitlines()
    indent = None
    block_lines: list[str] = []
    for ln in lines:
        if not ln.strip():
            block_lines.append("")
            continue
        leading = len(ln) - len(ln.lstrip(" "))
        if indent is None:
            indent = leading
        if leading < indent and ln.strip():
            break
        block_lines.append(ln[indent:] if leading >= indent else ln)
    # Trim trailing blank lines
    while block_lines and not block_lines[-1].strip():
        block_lines.pop()
    return "\n".join(block_lines)


def load_round_code(round_num: int) -> str:
    p = DATA_DIR / f"round_{round_num}" / "best" / "best_program.py"
    return read_text(p)


def load_round_info(round_num: int) -> dict:
    p = DATA_DIR / f"round_{round_num}" / "best" / "best_program_info.json"
    return read_json(p)


def load_round_config(round_num: int) -> str:
    p = DATA_DIR / f"config_round_{round_num}.yaml"
    if not p.exists():
        # config_round_N for the final round is sometimes not regenerated
        # because the run finished. Fall back to the previous round's config.
        for r in range(round_num - 1, 0, -1):
            alt = DATA_DIR / f"config_round_{r}.yaml"
            if alt.exists():
                return read_text(alt)
        return ""
    return read_text(p)


def main() -> None:
    log_text = read_text(LOG_PATH)
    log_sections = parse_log_sections(log_text)
    # Map log Round 0..N to folder round_1..round_(N+1)
    log_by_folder_round: dict[int, dict] = {}
    for sec in log_sections:
        folder_round = sec["round_num"] + 1
        log_by_folder_round[folder_round] = parse_round_body(sec["body"])

    rounds = []
    for r in range(1, NUM_ROUNDS + 1):
        info = load_round_info(r)
        code = load_round_code(r)
        config_yaml = load_round_config(r)
        sys_msg = extract_yaml_system_message(config_yaml)
        log_entry = log_by_folder_round.get(r, {})

        rounds.append({
            "round": r,
            "info": info,
            "code": code,
            "system_message": sys_msg,
            "log": log_entry,
        })

    out = {
        "title": "GeoRegime: Self-Evolving Spatial Regime Regionalization",
        "subtitle": (
            "Visualizing the evolutionary optimization of a spatial regime "
            "regionalization algorithm with RAG-guided knowledge retrieval "
            "across 10 rounds."
        ),
        "rounds": rounds,
    }

    out_path = WEB_DIR / "data.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    # Print a quick summary
    for rd in rounds:
        m = rd["info"]["metrics"]
        kn = (rd["log"].get("knowledge_needed") or {}) if rd["log"] else {}
        sq = kn.get("search_queries") if isinstance(kn, dict) else None
        print(
            f"round {rd['round']:2d}  randi={m['randi']:.4f}  ssr={m['ssr']:.2f}  "
            f"queries={len(sq) if sq else 0}  code_lines={len(rd['code'].splitlines())}"
        )


if __name__ == "__main__":
    main()
