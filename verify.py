#!/usr/bin/env python3
# verify.py -- Step 3: Verification script
#
# Two modes:
#   python verify.py gt   <gt.csv>   <output.txt>
#   python verify.py compare <out_a.txt> <out_b.txt>
#
# Mode 'gt':
#   For planted images: check found (x,y) == ground-truth and score == 0.
#   For unplanted images: report best score found.
#   Prints correct/wrong counts and PASS/FAIL.
#
# Mode 'compare':
#   Check two output files are identical line by line.
#   Prints first mismatch (if any) and PASS/FAIL.

import sys, csv

def load_output(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_id = int(parts[0])
            rows[img_id] = (int(parts[1]), int(parts[2]), int(parts[3]))
    return rows

def mode_gt(gt_path, out_path):
    # Load ground truth
    gt = {}
    with open(gt_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id  = int(row['image_id'])
            planted = int(row['planted']) == 1
            x       = int(row['x'])
            y       = int(row['y'])
            gt[img_id] = (planted, x, y)

    pred = load_output(out_path)
    N = len(gt)
    assert len(pred) == N, f'GT has {N} images but output has {len(pred)}'

    correct   = 0   # planted and (x,y) matches exactly
    wrong_pos = 0   # planted but wrong (x,y)
    n_planted = sum(1 for p, x, y in gt.values() if p)

    for img_id in range(N):
        gt_planted, gt_x, gt_y = gt[img_id]
        px, py, sc = pred[img_id]

        if gt_planted:
            if px == gt_x and py == gt_y:
                assert sc == 0, f'Image {img_id}: planted but score={sc} != 0'
                correct += 1
            else:
                wrong_pos += 1
                print(f'  WRONG  img {img_id}: expected ({gt_x},{gt_y}) got ({px},{py}) score={sc}')
        else:
            # Unplanted: just report if a very low score was found (informational)
            if sc == 0:
                print(f'  WARN   img {img_id}: unplanted but score=0 at ({px},{py})')

    print(f'GT check: {correct}/{n_planted} planted correctly detected, {wrong_pos} wrong positions')
    if wrong_pos == 0 and correct == n_planted:
        print('PASS')
    else:
        print('FAIL')

def mode_compare(a_path, b_path):
    with open(a_path) as fa, open(b_path) as fb:
        lines_a = [l.rstrip() for l in fa if l.strip()]
        lines_b = [l.rstrip() for l in fb if l.strip()]

    if len(lines_a) != len(lines_b):
        print(f'FAIL: line count differs ({len(lines_a)} vs {len(lines_b)})')
        return

    mismatch = 0
    for i, (la, lb) in enumerate(zip(lines_a, lines_b)):
        if la != lb:
            if mismatch == 0:
                print(f'First mismatch at line {i+1}:')
                print(f'  A: {la}')
                print(f'  B: {lb}')
            mismatch += 1

    if mismatch == 0:
        print(f'Files identical ({len(lines_a)} lines). PASS')
    else:
        print(f'FAIL: {mismatch} mismatches total')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: verify.py gt <gt.csv> <output.txt>')
        print('       verify.py compare <out_a.txt> <out_b.txt>')
        sys.exit(1)
    mode = sys.argv[1]
    if mode == 'gt':
        mode_gt(sys.argv[2], sys.argv[3])
    elif mode == 'compare':
        mode_compare(sys.argv[2], sys.argv[3])
    else:
        print(f'Unknown mode: {mode}')
        sys.exit(1)
