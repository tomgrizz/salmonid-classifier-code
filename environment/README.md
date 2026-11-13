# Environments

Three Python environments were used. Only the first is needed to check the paper's numbers.

| File | What it is for | Needed for |
|---|---|---|
| `tier1.lock.txt` | The exact versions (15 packages, including indirect ones) behind every published table and figure that is recomputed from `data/`. | Tier 1: `python run_tier1.py` |
| `track.lock.txt` | The base environment that tracked fish and built the counts. | Tier 2: tracking, counting, photo figures, map |
| `deim.lock.txt` | The GPU environment that trained the model and ran detection (`pip freeze`, 78 packages). | Tier 2: training, detection |

All three were recorded on the original workstation: Python 3.11.9, Windows 11, NVIDIA RTX
5090, CUDA 12.8.

The shorter `requirements*.txt` files in the repository root name only the packages the code
imports directly, with minimum versions. Use a lock file when you need the same versions,
for example to get figure files that are byte-identical to the published ones.

Two things worth knowing:

- **Tracking library version.** The published counts used `boxmot` 10.0.52. Version 21 cannot
  run this code. Version 10.0.52 declares an old numpy but works with numpy 2.4.4, which is
  what it ran with here; install it with `pip install --no-deps boxmot==10.0.52`.
- **Why two environments for Tier 2.** The DEIMv2 code and the tracking library were developed
  against different versions of the same underlying packages, so detection (Pass A) and
  tracking (Pass B) run as separate steps in separate environments and pass results through
  files on disk.
