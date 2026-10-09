#!/usr/bin/env python3
"""
evaluate.py
-----------
Compares matcher output against ground-truth CSV.

Usage
-----
  python evaluate.py <output.txt> <gt_N<N>.csv>

Output format expected (one line per image)
-------------------------------------------
  image_id x y score

Ground-truth CSV columns
------------------------
  image_id, planted, x, y   (x=y=-1 if not planted)

Metrics reported
----------------
  - Total images
  - Planted images (from GT)
  - Correctly detected   (planted and (x,y) matches exactly)
  - False positives      (not planted but reported score = 0 or very low)
  - Correctly rejected   (not planted, score > 0)
  - Precision, Recall, F1  (for planted detection)
  - Wrong position count   (planted but wrong x,y)
"""

import sys, csv

def main():
    if len(sys.argv) < 3:
        print("Usage: evaluate.py <output.txt> <gt_N<N>.csv>")
        sys.exit(1)

    out_file = sys.argv[1]
    gt_file  = sys.argv[2]

    # ---- load ground truth --------------------------------------------------
    gt = {}   # image_id -> (planted:bool, x:int, y:int)
    with open(gt_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id  = int(row["image_id"])
            planted = int(row["planted"]) == 1
            x       = int(row["x"])
            y       = int(row["y"])
            gt[img_id] = (planted, x, y)

    # ---- load matcher output ------------------------------------------------
    pred = {}  # image_id -> (x:int, y:int, score:int)
    with open(out_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_id = int(parts[0])
            px     = int(parts[1])
            py     = int(parts[2])
            score  = int(parts[3])
            pred[img_id] = (px, py, score)

    N = len(gt)
    assert len(pred) == N, f"GT has {N} images but output has {len(pred)}"

    # ---- statistics ---------------------------------------------------------
    planted_ids  = {i for i, (p, x, y) in gt.items() if p}
    n_planted    = len(planted_ids)
    n_unplanted  = N - n_planted

    correct_detect = 0    # planted, correct (x,y)
    wrong_pos      = 0    # planted, wrong (x,y)
    false_positive = 0    # not planted but score == 0
    correct_reject = 0    # not planted and score > 0

    for img_id in range(N):
        gt_planted, gt_x, gt_y = gt[img_id]
        px, py, sc = pred[img_id]

        if gt_planted:
            if px == gt_x and py == gt_y:
                correct_detect += 1
            else:
                wrong_pos += 1
        else:
            if sc == 0:
                false_positive += 1
            else:
                correct_reject += 1

    precision = correct_detect / (correct_detect + false_positive) \
                if (correct_detect + false_positive) > 0 else float("nan")
    recall    = correct_detect / n_planted \
                if n_planted > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) \
         if (precision + recall) > 0 else float("nan")

    # ---- report -------------------------------------------------------------
    print(f"{'='*55}")
    print(f"  Evaluation: {out_file}")
    print(f"  Ground truth: {gt_file}")
    print(f"{'='*55}")
    print(f"  Total images          : {N}")
    print(f"  Planted               : {n_planted}  ({100*n_planted/N:.1f}%)")
    print(f"  Correctly detected    : {correct_detect}")
    print(f"  Wrong position        : {wrong_pos}")
    print(f"  False positives       : {false_positive}")
    print(f"  Correctly rejected    : {correct_reject}")
    print(f"  Precision             : {precision:.4f}")
    print(f"  Recall                : {recall:.4f}")
    print(f"  F1 score              : {f1:.4f}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
