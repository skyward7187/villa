#!/usr/bin/env python3
"""Reduce the placed patches to a mutually consistent core.

Necessary conditions, not sufficient ones: passing every test here means no
evidence contradicts the patch, not that the patch is proven right.

Tiers, each strictly inside the last:
  0  placed      every patch at least one network fiber runs along
  1  unstraddled drop patches whose own fibers disagree about the winding
  2  no-B        drop patches a fiber crosses in 2D but misses in 3D
  3  A-consistent  greedily drop patches until no two overlapping patches
                   disagree about the winding
  4  corroborated  keep only what a second independent source confirms:
                   >= 2 fibers, or an agreeing 3D-overlapping neighbour
"""

import json
from collections import defaultdict
from pathlib import Path

import common

d = json.load(open(common.consistency_report()))
allp = d['all_patches']
nfib = d['fiber_count']
straddled = set(d['straddled'])
Bv = d['B_violations']
Cv = d['C_violations']
Adis = {k: set(v) for k, v in d['A_disagree'].items()}
pairs = [tuple(p) for p in d['overlap_pairs']]

agree_adj = defaultdict(set)
for a, b in pairs:
    if b not in Adis.get(a, ()):
        agree_adj[a].add(b)
        agree_adj[b].add(a)

print(f'tier 0  placed          {len(allp):5d}')

t1 = [p for p in allp if p not in straddled]
print(f'tier 1  unstraddled     {len(t1):5d}   (-{len(allp) - len(t1)} straddle a wrap)')

t2 = [p for p in t1 if p not in Bv]
print(f'tier 2  no-B            {len(t2):5d}   (-{len(t1) - len(t2)} crossed by a fiber they miss)')

# tier 3: greedy minimum-ish vertex cover over the A-conflict graph
alive = set(t2)
conf = {p: Adis.get(p, set()) & alive for p in alive}
removed_A = []
while True:
    worst, deg = None, 0
    for p in alive:
        n = len(conf[p])
        if n > deg or (n == deg and n > 0 and worst is not None
                       and (nfib[p], p) < (nfib[worst], worst)):
            worst, deg = p, n
    if not worst or deg == 0:
        break
    alive.discard(worst)
    removed_A.append(worst)
    for q in conf.pop(worst):
        conf[q].discard(worst)
t3 = sorted(alive)
print(f'tier 3  A-consistent    {len(t3):5d}   (-{len(removed_A)} to break every '
      f'winding disagreement)')

t4 = [p for p in t3 if nfib[p] >= 2 or (agree_adj[p] & alive)]
print(f'tier 4  corroborated    {len(t4):5d}   (-{len(t3) - len(t4)} rest on a single '
      f'fiber with no agreeing neighbour)')

t4c = [p for p in t4 if p not in Cv]
print(f'        + C-clean       {len(t4c):5d}   (-{len(t4) - len(t4c)} also have a '
      f'footprint clash; C is the weakest test)')

json.dump({
    'description': 'consistency tiers for the patches carrying fiber network 1',
    'note': ('these are necessary conditions, not proof of correctness; '
             'tier 4 is the recommended trusted core'),
    'tiers': {
        'placed': allp,
        'unstraddled': t1,
        'no_B': t2,
        'A_consistent': t3,
        'corroborated': t4,
        'corroborated_C_clean': t4c,
    },
    'counts': {k: v for k, v in [
        ('placed', len(allp)), ('unstraddled', len(t1)), ('no_B', len(t2)),
        ('A_consistent', len(t3)), ('corroborated', len(t4)),
        ('corroborated_C_clean', len(t4c))]},
    'flags': {p: {
        'fibers': nfib[p],
        'straddled': p in straddled,
        'B': Bv.get(p, []),
        'A_disagree': sorted(Adis.get(p, ())),
        'C': Cv.get(p, []),
    } for p in allp if (p in straddled or p in Bv or p in Adis or p in Cv)},
}, open(common.trust_report(), 'w'), indent=1)

# A flat per-patch verdict table, easier to sort through than the JSON.
import csv

sets = {k: set(v) for k, v in {
    'unstraddled': t1, 'no_B': t2, 'A_consistent': t3, 'corroborated': t4}.items()}
reasons = [('unstraddled', 'drop:straddles a wrap'),
           ('no_B', 'drop:fiber crosses but misses'),
           ('A_consistent', 'drop:winding disagreement'),
           ('corroborated', 'drop:uncorroborated')]
core, clean = set(t4), set(t4c)


def verdict(p):
    if p in clean:
        return 'core+Cclean'
    if p in core:
        return 'core'
    for key, label in reasons:
        if p not in sets[key]:
            return label
    return 'core'


csv_path = common.trust_report().with_suffix('.csv')
with open(csv_path, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['patch', 'verdict', 'fibers', 'straddled_fibers',
                'B_violations', 'A_disagreements', 'C_clashes'])
    for p in allp:
        f = {'fibers': nfib[p], 'straddled': p in straddled,
             'B': Bv.get(p, []), 'A': Adis.get(p, ()), 'C': Cv.get(p, [])}
        w.writerow([p, verdict(p), f['fibers'], 1 if f['straddled'] else 0,
                    len(f['B']), len(f['A']), len(f['C'])])
print(f'wrote {csv_path}')
print(f'\nwrote {common.trust_report()}')
