#!/usr/bin/env python3
"""
exp4_multirater.py
==================
Adds a second rater to the relevance labels and computes inter-rater agreement,
addressing the author-only-evaluation limitation.

Two-step workflow
-----------------
1) Generate a blank template for rater 2 (same papers, grades zeroed):

       python exp4_multirater.py --init

   This reads relevance_labels.json (rater 1 = you) and writes
   relevance_labels_rater2.json with the same papers per query. Have a second
   person (labmate) assign grades 0-3 independently, WITHOUT seeing your grades.

2) Compute agreement once rater 2 has filled it in:

       python exp4_multirater.py

   Reports, per query and overall:
     - percent exact agreement on graded scale (0-3)
     - percent agreement on binary relevance (grade>0 vs grade=0)
     - Cohen's kappa on binary relevance

Notes
-----
- Cohen's kappa corrects for chance agreement; > 0.6 is substantial,
  > 0.8 is near-perfect.
- Only papers graded by BOTH raters for a query are included in that query's
  agreement computation.
"""

import os
import json

LABELS_FILE = "relevance_labels.json"
RATER2_FILE = "relevance_labels_rater2.json"


def init_rater2():
    if not os.path.exists(LABELS_FILE):
        print(f"ERROR: {LABELS_FILE} not found. Create rater-1 labels first.")
        return
    with open(LABELS_FILE) as f:
        r1 = json.load(f)

    r2 = {
        "_instructions": ("RATER 2: assign a relevance grade to each paper per "
                          "query WITHOUT looking at rater 1's grades. "
                          "3=highly relevant, 2=relevant, 1=marginal, 0=not relevant."),
        "queries": {},
    }
    for run_file, entry in r1["queries"].items():
        r2["queries"][run_file] = {
            "query": entry["query"],
            "relevance": {pid: 0 for pid in entry["relevance"].keys()},
        }
    with open(RATER2_FILE, "w") as f:
        json.dump(r2, f, indent=2)
    print(f"Wrote rater-2 template → {RATER2_FILE}")
    print("Have a second person grade it independently, then run: python exp4_multirater.py")


def cohens_kappa_binary(a, b):
    """a, b: lists of 0/1 labels of equal length."""
    n = len(a)
    if n == 0:
        return None
    # observed agreement
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # expected agreement
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def compute():
    if not (os.path.exists(LABELS_FILE) and os.path.exists(RATER2_FILE)):
        print(f"ERROR: need both {LABELS_FILE} and {RATER2_FILE}. "
              f"Run --init first, then have rater 2 grade.")
        return

    with open(LABELS_FILE) as f:
        r1 = json.load(f)
    with open(RATER2_FILE) as f:
        r2 = json.load(f)

    print("=" * 78)
    print("  EXPERIMENT 4: INTER-RATER AGREEMENT")
    print("=" * 78)
    hdr = f"{'query':30}{'n':>5}{'exact%':>9}{'binary%':>9}{'kappa':>8}"
    print(hdr); print("-" * len(hdr))

    all_g1_bin, all_g2_bin = [], []
    all_exact = []
    for run_file, e1 in r1["queries"].items():
        if run_file not in r2["queries"]:
            continue
        rel1 = e1["relevance"]
        rel2 = r2["queries"][run_file]["relevance"]
        shared = [pid for pid in rel1 if pid in rel2]
        if not shared:
            continue

        g1 = [int(rel1[pid]) for pid in shared]
        g2 = [int(rel2[pid]) for pid in shared]
        g1b = [1 if x > 0 else 0 for x in g1]
        g2b = [1 if x > 0 else 0 for x in g2]

        exact = sum(1 for x, y in zip(g1, g2) if x == y) / len(shared)
        binary = sum(1 for x, y in zip(g1b, g2b) if x == y) / len(shared)
        kappa = cohens_kappa_binary(g1b, g2b)

        all_exact.extend([1 if x == y else 0 for x, y in zip(g1, g2)])
        all_g1_bin.extend(g1b)
        all_g2_bin.extend(g2b)

        kstr = f"{kappa:.3f}" if kappa is not None else "n/a"
        print(f"{e1['query'][:30]:30}{len(shared):>5}{exact*100:>8.1f}%"
              f"{binary*100:>8.1f}%{kstr:>8}")

    print("-" * len(hdr))
    overall_exact = sum(all_exact) / len(all_exact) if all_exact else 0
    overall_bin = sum(1 for x, y in zip(all_g1_bin, all_g2_bin) if x == y) / len(all_g1_bin) \
        if all_g1_bin else 0
    overall_kappa = cohens_kappa_binary(all_g1_bin, all_g2_bin)
    kstr = f"{overall_kappa:.3f}" if overall_kappa is not None else "n/a"
    print(f"{'OVERALL':30}{len(all_g1_bin):>5}{overall_exact*100:>8.1f}%"
          f"{overall_bin*100:>8.1f}%{kstr:>8}")

    interp = ("near-perfect" if overall_kappa and overall_kappa > 0.8 else
              "substantial" if overall_kappa and overall_kappa > 0.6 else
              "moderate" if overall_kappa and overall_kappa > 0.4 else "fair/poor")
    print(f"\n  Cohen's kappa (binary relevance): {kstr} ({interp})")

    out = {
        "overall": {
            "n_judgments": len(all_g1_bin),
            "exact_agreement": overall_exact,
            "binary_agreement": overall_bin,
            "cohens_kappa_binary": overall_kappa,
            "interpretation": interp,
        }
    }
    os.makedirs("results", exist_ok=True)
    with open("results/multirater_agreement.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → results/multirater_agreement.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        init_rater2()
    else:
        compute()
