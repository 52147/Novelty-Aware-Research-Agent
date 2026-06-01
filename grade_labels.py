#!/usr/bin/env python3
"""
grade_labels.py — fills relevance_labels.json with author-assigned grades.
Edit the GRADES dict to your judgment, then run:  python grade_labels.py
Re-run ir_metrics.py afterward.

Grades: 3=highly relevant, 2=relevant, 1=marginal, 0=not relevant.
Includes papers BEYOND the retrieved top-5 so Recall@5 is meaningful.
"""
import json

LABELS_FILE = "relevance_labels.json"

# ---- EDIT THESE TO YOUR JUDGMENT --------------------------------------------
GRADES = {
  "run_01.json": {   # Compare multi-agent LLM frameworks for collaborative reasoning
    # retrieved top-5
    "agentverse_2023": 3, "autogen_2023": 3, "agentsurvey_2023": 2,
    "agentbench_2023": 1, "openagents_2023": 1,
    # other relevant papers in corpus (not retrieved) -> for Recall
    "metagpt_2023": 3, "camel_2023": 3, "chatdev_2023": 2,
    "multidebate_2023": 2, "metaagents_2023": 2, "roco_2023": 1,
  },
  "run_02.json": {   # What evaluation methods exist for LLM reasoning agents?
    "agentbench_2023": 3, "agentsurvey_2023": 2, "art_2023": 1,
    "cot_2022": 1, "rap_2023": 1,
    # other evaluation/benchmark papers -> for Recall
    "mint_2023": 3, "gaia_2023": 3, "webarena_2023": 2,
    "swebench_2023": 2, "toolbench_2023": 2,
  },
  "run_03.json": {   # Compare verbal reinforcement and role-playing approaches
    "reflexion_2023": 3, "camel_2023": 3, "agentverse_2023": 2,
    "generativeagents_2023": 2, "innermono_2022": 1,
    # other related papers -> for Recall
    "expel_2023": 2, "chatdev_2023": 1, "metaagents_2023": 1,
  },
}
# -----------------------------------------------------------------------------

with open(LABELS_FILE) as f:
    labels = json.load(f)

for rf, grades in GRADES.items():
    if rf not in labels["queries"]:
        print(f"WARNING: {rf} not in labels file, skipping")
        continue
    rel = labels["queries"][rf].setdefault("relevance", {})
    for pid, g in grades.items():
        rel[pid] = g

with open(LABELS_FILE, "w") as f:
    json.dump(labels, f, indent=2)

print("Graded. Per-query relevant-paper counts:")
for rf, grades in GRADES.items():
    n_rel = sum(1 for g in grades.values() if g > 0)
    print(f"  {rf}: {len(grades)} graded, {n_rel} relevant (grade>0)")
print("\nNow run:  python ir_metrics.py")
