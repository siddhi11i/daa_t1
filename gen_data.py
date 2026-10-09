#!/usr/bin/env python3
# gen_data.py -- Step 1: Dataset generation for Parallel Template Matching
import os, csv, random
import numpy as np
from PIL import Image

SRC_DIR = 'val2017'
IMG_H = IMG_W = 256
TMPL_H = TMPL_W = 32
PLANT_RATE = 0.30
SEED = 42
SIZES = [1000, 5000, 10000, 50000]

print('Loading COCO val2017 images ...', flush=True)
jpgs = sorted(f for f in os.listdir(SRC_DIR) if f.lower().endswith('.jpg'))
assert len(jpgs) >= 5000
jpgs = jpgs[:5000]
base_imgs = []
for i, fname in enumerate(jpgs):
    img = Image.open(os.path.join(SRC_DIR, fname)).convert('L')
    img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
    base_imgs.append(np.array(img, dtype=np.uint8))
    if (i+1) % 500 == 0:
        print(f'  loaded {i+1}/5000', flush=True)
print('All 5000 loaded.', flush=True)

# Blocky 32x32 template: 4x4 solid pixel blocks
rng_tmpl = np.random.default_rng(SEED)
BLOCK = 4
template = np.zeros((TMPL_H, TMPL_W), dtype=np.uint8)
for by in range(TMPL_H // BLOCK):
    for bx in range(TMPL_W // BLOCK):
        template[by*BLOCK:(by+1)*BLOCK, bx*BLOCK:(bx+1)*BLOCK] = int(rng_tmpl.integers(0, 256))
with open('template.bin', 'wb') as f:
    f.write(template.flatten().tobytes())
print('Wrote template.bin', flush=True)

all_imgs = list(base_imgs)
for img in base_imgs:
    all_imgs.append(np.fliplr(img).copy())
rng_n = np.random.default_rng(SEED + 1)
for i in range(40000):
    src = all_imgs[i % 10000]
    noisy = np.clip(src.astype(np.int16) + rng_n.normal(0, 15, src.shape), 0, 255).astype(np.uint8)
    all_imgs.append(noisy)
print(f'Pool: {len(all_imgs)}', flush=True)

def plant_tmpl(img, tmpl, x, y):
    img[y:y+tmpl.shape[0], x:x+tmpl.shape[1]] = tmpl

for N in SIZES:
    print(f'Generating N={N} ...', flush=True)
    mmap = np.memmap(f'images_N{N}.bin', dtype=np.uint8, mode='w+', shape=(N, IMG_H, IMG_W))
    rng_p = random.Random(SEED + N)
    gt_rows = []
    for img_id in range(N):
        img = all_imgs[img_id].copy()
        planted = rng_p.random() < PLANT_RATE
        if planted:
            px = rng_p.randint(0, IMG_W - TMPL_W)
            py = rng_p.randint(0, IMG_H - TMPL_H)
            plant_tmpl(img, template, px, py)
            gt_rows.append((img_id, 1, px, py))
        else:
            gt_rows.append((img_id, 0, -1, -1))
        mmap[img_id] = img
        if (img_id+1) % 5000 == 0:
            print(f'  {img_id+1}/{N}', flush=True)
    del mmap
    sz = os.path.getsize(f'images_N{N}.bin') // 1048576
    print(f'  images_N{N}.bin written ({sz} MB)')
    with open(f'gt_N{N}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_id','planted','x','y'])
        w.writerows(gt_rows)
    pc = sum(1 for r in gt_rows if r[1])
    print(f'  gt_N{N}.csv written ({pc} planted / {N})')

print('Done.')
