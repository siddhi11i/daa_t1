#!/usr/bin/env python3
# plot.py -- Step 6: Read results.csv, compute speedup/efficiency, save plots
#
# Requires: pandas, matplotlib

import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV = 'results.csv'
df  = pd.read_csv(CSV)

# Median compute time per configuration
med = df.groupby(['version', 'N', 'threads_or_procs'])['compute_time'].median().reset_index()
med.columns = ['version', 'N', 'tp', 'median_compute']

SIZES = sorted(df['N'].unique())
COLORS = {'seq': '#555', 'omp': '#1f77b4', 'mpi': '#d62728'}
MARKERS = {'seq': 'o', 'omp': 's', 'mpi': '^'}

# ---- 1. Compute time vs threads/procs (one subplot per N) ------------------
fig, axes = plt.subplots(1, len(SIZES), figsize=(5*len(SIZES), 4), sharey=False)
for ax, N in zip(axes, SIZES):
    for ver in ['omp', 'mpi']:
        sub = med[(med['version'] == ver) & (med['N'] == N)]
        if sub.empty: continue
        ax.plot(sub['tp'], sub['median_compute'],
                marker=MARKERS[ver], label=ver, color=COLORS[ver])
    seq_t = med[(med['version'] == 'seq') & (med['N'] == N) & (med['tp'] == 1)]
    if not seq_t.empty:
        ax.axhline(seq_t['median_compute'].values[0], ls='--', color=COLORS['seq'], label='seq')
    ax.set_title(f'N={N}')
    ax.set_xlabel('Threads / Procs')
    ax.set_ylabel('Median compute time (s)')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
plt.tight_layout()
plt.savefig('plot_time_vs_threads.png', dpi=120)
plt.close()
print('Saved plot_time_vs_threads.png')

# ---- 2. Speedup vs threads/procs ------------------------------------------
fig, axes = plt.subplots(1, len(SIZES), figsize=(5*len(SIZES), 4), sharey=False)
for ax, N in zip(axes, SIZES):
    seq_t = med[(med['version'] == 'seq') & (med['N'] == N) & (med['tp'] == 1)]
    if seq_t.empty: continue
    t_seq = seq_t['median_compute'].values[0]

    for ver in ['omp', 'mpi']:
        sub = med[(med['version'] == ver) & (med['N'] == N)]
        if sub.empty: continue
        speedup = t_seq / sub['median_compute']
        ax.plot(sub['tp'], speedup,
                marker=MARKERS[ver], label=ver, color=COLORS[ver])

    # Ideal speedup line
    max_tp = med[med['N'] == N]['tp'].max()
    tp_range = range(1, int(max_tp)+1)
    ax.plot(tp_range, tp_range, 'k--', label='ideal', lw=1)
    ax.set_title(f'N={N}')
    ax.set_xlabel('Threads / Procs')
    ax.set_ylabel('Speedup')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
plt.tight_layout()
plt.savefig('plot_speedup.png', dpi=120)
plt.close()
print('Saved plot_speedup.png')

# ---- 3. Efficiency vs threads/procs ----------------------------------------
fig, axes = plt.subplots(1, len(SIZES), figsize=(5*len(SIZES), 4), sharey=False)
for ax, N in zip(axes, SIZES):
    seq_t = med[(med['version'] == 'seq') & (med['N'] == N) & (med['tp'] == 1)]
    if seq_t.empty: continue
    t_seq = seq_t['median_compute'].values[0]

    for ver in ['omp', 'mpi']:
        sub = med[(med['version'] == ver) & (med['N'] == N)]
        if sub.empty: continue
        efficiency = (t_seq / sub['median_compute']) / sub['tp']
        ax.plot(sub['tp'], efficiency,
                marker=MARKERS[ver], label=ver, color=COLORS[ver])

    ax.axhline(1.0, ls='--', color='k', label='ideal', lw=1)
    ax.set_title(f'N={N}')
    ax.set_xlabel('Threads / Procs')
    ax.set_ylabel('Efficiency')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
plt.tight_layout()
plt.savefig('plot_efficiency.png', dpi=120)
plt.close()
print('Saved plot_efficiency.png')

# ---- 4. Time vs N (one line per version+tp combo) --------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for ver in ['seq', 'omp', 'mpi']:
    sub = med[med['version'] == ver]
    for tp in sorted(sub['tp'].unique()):
        ts = sub[sub['tp'] == tp].sort_values('N')
        label = f'{ver}' if tp == 1 else f'{ver}-{tp}'
        ax.plot(ts['N'], ts['median_compute'],
                marker=MARKERS[ver], label=label, color=COLORS[ver],
                alpha=0.5 + 0.5/max(1, int(tp)/8))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('N (images)')
ax.set_ylabel('Median compute time (s)')
ax.set_title('Compute time vs N')
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig('plot_time_vs_N.png', dpi=120)
plt.close()
print('Saved plot_time_vs_N.png')
