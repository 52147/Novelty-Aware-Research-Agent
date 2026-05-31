

#!/usr/bin/env python3
"""
validate_gap_matrix.py

Purpose
-------
Create a lightweight validation set for the deterministic gap matrix.

Workflow
--------
1) Generate a validation form:

    python validate_gap_matrix.py --init

This writes:

    gap_validation_form.csv

2) Open gap_validation_form.csv and fill the `label` column:

    plausible      = valid corpus-level absence signal
    too_broad      = too generic / not specific enough
    indirect       = only weakly implied by the matrix
    meaningless    = not a useful gap

Optional: fill `notes`.

3) Compute validation precision:

    python validate_gap_matrix.py

Outputs:

    results/gap_validation_metrics.json

Notes
-----
This does NOT claim the gap exists in the whole literature.
It only validates whether empty problem-method cells are plausible
corpus-level gap candidates within the retrieved corpus.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter

MATRIX_CSV = "gap_matrix.csv"
FORM_CSV = "gap_validation_form.csv"
RESULTS_DIR = "results"
OUT_JSON = os.path.join(RESULTS_DIR, "gap_validation_metrics.json")

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 42
VALID_LABELS = {"plausible", "too_broad", "indirect", "meaningless"}
POSITIVE_LABEL = "plausible"

DEFAULT_GAP_MATRIX = [
    [
        "Problem / Method",
        "Benchmarking",
        "Conversational multi-agent framework",
        "Decoupled reasoning from observation",
        "Memory + reflection + planning",
        "Multi-agent collaboration framework",
        "ReAct reasoning-action loop",
        "Reasoning prompting",
        "Role-based multi-agent framework",
        "Role-playing / inception prompting",
        "Simulation-based multi-agent coordination",
        "Survey and taxonomy",
        "Verbal reflection",
    ],
    ["Agent evaluation", "agentbench_2023", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"],
    ["Agent reasoning", "—", "—", "—", "—", "—", "—", "cot_2022", "—", "—", "—", "—", "—"],
    ["Agent self-improvement", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "reflexion_2023"],
    ["Agent survey / taxonomy", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "agentsurvey_2023", "—"],
    ["Human behavior simulation", "—", "—", "—", "generativeagents_2023", "—", "—", "—", "—", "—", "—", "—", "—"],
    ["Multi-agent coordination", "—", "autogen_2023", "—", "—", "agentverse_2023", "—", "—", "metagpt_2023", "camel_2023", "metaagents_2023", "—", "—"],
    ["Reasoning-action efficiency", "—", "—", "rewoo_2023", "—", "—", "—", "—", "—", "—", "—", "—", "—"],
    ["Reasoning-action interaction", "—", "—", "—", "—", "—", "react_2022", "—", "—", "—", "—", "—", "—"],
]


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# Write the default deterministic gap matrix if missing
def write_default_gap_matrix(path=MATRIX_CSV):
    """Write the paper's deterministic gap matrix when no CSV exists yet."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(DEFAULT_GAP_MATRIX)
    print(f"Default deterministic gap matrix not found; wrote → {path}")


def load_gap_matrix(path=MATRIX_CSV):
    if not os.path.exists(path):
        if path == MATRIX_CSV:
            write_default_gap_matrix(path)
        else:
            raise FileNotFoundError(
                f"Cannot find {path}. Put your deterministic gap matrix CSV at this path, "
                f"or run with --matrix path/to/file.csv"
            )

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows[0]) < 2:
        raise ValueError("Gap matrix CSV appears empty or malformed.")

    header = rows[0]
    method_cols = header[1:]
    matrix_rows = []

    for row in rows[1:]:
        if not row:
            continue
        problem = row[0].strip()
        values = row[1:]
        if not problem:
            continue
        # Pad if row is short.
        if len(values) < len(method_cols):
            values += [""] * (len(method_cols) - len(values))
        matrix_rows.append((problem, values[: len(method_cols)]))

    return method_cols, matrix_rows


def is_empty_cell(value):
    v = (value or "").strip()
    return v == "" or v == "—" or v == "-" or v.lower() in {"na", "n/a", "none", "null"}


def collect_empty_cells(method_cols, matrix_rows):
    empty_cells = []
    filled_cells = []

    for problem, values in matrix_rows:
        for method, value in zip(method_cols, values):
            cell = {
                "problem": problem,
                "method": method,
                "cell_value": (value or "").strip(),
            }
            if is_empty_cell(value):
                empty_cells.append(cell)
            else:
                filled_cells.append(cell)

    return empty_cells, filled_cells


def init_form(matrix_path, sample_size, seed):
    method_cols, matrix_rows = load_gap_matrix(matrix_path)
    empty_cells, filled_cells = collect_empty_cells(method_cols, matrix_rows)

    if not empty_cells:
        raise ValueError("No empty cells found in the matrix.")

    random.seed(seed)
    sample_size = min(sample_size, len(empty_cells))
    sampled = random.sample(empty_cells, sample_size)

    with open(FORM_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "problem",
                "missing_method",
                "label",
                "notes",
            ],
        )
        writer.writeheader()
        for i, cell in enumerate(sampled, start=1):
            writer.writerow(
                {
                    "id": i,
                    "problem": cell["problem"],
                    "missing_method": cell["method"],
                    "label": "",
                    "notes": "",
                }
            )

    print(f"Wrote validation form → {FORM_CSV}")
    print(f"Sampled {sample_size} empty cells out of {len(empty_cells)} total empty cells.")
    print("Fill the label column with: plausible / too_broad / indirect / meaningless")
    print("Then run: python validate_gap_matrix.py")


def load_completed_form(path=FORM_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}. Run `python validate_gap_matrix.py --init` first.")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} has no rows.")

    return rows


def compute_metrics(form_path):
    ensure_results_dir()
    rows = load_completed_form(form_path)

    labeled = []
    unlabeled = []
    invalid = []

    for row in rows:
        label = (row.get("label") or "").strip().lower()
        if not label:
            unlabeled.append(row)
        elif label not in VALID_LABELS:
            invalid.append(row)
        else:
            labeled.append({**row, "label": label})

    if invalid:
        bad = sorted(set((r.get("label") or "").strip() for r in invalid))
        raise ValueError(f"Invalid labels found: {bad}. Use only: {sorted(VALID_LABELS)}")

    if not labeled:
        raise ValueError("No labeled rows found. Fill the label column first.")

    counts = Counter(row["label"] for row in labeled)
    n = len(labeled)
    plausible = counts[POSITIVE_LABEL]
    precision = plausible / n if n else 0.0

    out = {
        "n_sampled": len(rows),
        "n_labeled": n,
        "n_unlabeled": len(unlabeled),
        "label_counts": dict(counts),
        "positive_label": POSITIVE_LABEL,
        "gap_precision": precision,
        "interpretation": (
            "Fraction of sampled empty cells judged plausible as corpus-level gap candidates. "
            "This is not a claim about the broader literature."
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("GAP MATRIX VALIDATION")
    print("=" * 72)
    print(f"Sampled rows:      {len(rows)}")
    print(f"Labeled rows:      {n}")
    print(f"Unlabeled rows:    {len(unlabeled)}")
    print(f"Plausible gaps:    {plausible}")
    print(f"Gap precision:     {precision:.3f}")
    print("\nLabel counts:")
    for label in sorted(VALID_LABELS):
        print(f"  {label:12s} {counts[label]}")
    print(f"\nSaved → {OUT_JSON}")

    print("\nLaTeX table:")
    print(r"\begin{table}[t]")
    print(r"\caption{Validation of sampled deterministic gap-matrix empty cells.}")
    print(r"\label{tab:gap-validation}")
    print(r"\begin{tabular}{lr}")
    print(r"\toprule")
    print(r"\textbf{Metric} & \textbf{Value} \\")
    print(r"\midrule")
    print(f"Sampled empty cells & {len(rows)} \\")
    print(f"Labeled cells & {n} \\")
    print(f"Plausible cells & {plausible} \\")
    print(f"Gap precision & {precision:.3f} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Generate gap_validation_form.csv")
    parser.add_argument("--matrix", default=MATRIX_CSV, help="Path to deterministic gap matrix CSV")
    parser.add_argument("--form", default=FORM_CSV, help="Path to completed validation form CSV")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.init:
        init_form(args.matrix, args.sample_size, args.seed)
    else:
        compute_metrics(args.form)


if __name__ == "__main__":
    main()