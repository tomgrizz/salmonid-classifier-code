# DEIMv2 external-repo pin

The single-stage detector trains with the external **DEIMv2** repo
(`Intellindust-AI-Lab/DEIMv2`, Apache-2.0), cloned at
`C:\Users\lomu-\Documents\GitHub\DEIMv2` — OUTSIDE this git repo. To reproduce a run you need the
exact DEIMv2 state below, plus the fish configs (snapshotted in `single_stage/configs/`).

## Pinned state

- **Commit:** `0fff8d4` (full SHA in `DEIMV2_SHA.txt`).
- **Working-tree patch:** `deimv2_worktree.patch` — the DEIMv2 clone is **dirty**, and the patch is
  REQUIRED (not optional cleanup). It contains two load-bearing edits to
  `engine/data/transforms/_transforms.py`:
  1. **Registers `ColorJitter`** as a DEIMv2 transform — the de-chrome augmentation. The v2.2+ fish
     recipe uses `type: ColorJitter` *instead of* `RandomPhotometricDistort`, whose RGB channel
     permutation destroys the color cue that separates chrome Coho / Rainbow / Chinook. The v25
     config references `ColorJitter`, so **without this patch the config fails to build.**
  2. **torchvision ≥0.22 rename** `_transform` → `transform` on `PadToSize`, `ConvertBoxes`,
     `ConvertPILImage` (the beta-transforms API changed).

## Reconstruct

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIMv2
cd DEIMv2 && git checkout 0fff8d4
git apply /path/to/single_stage/deimv2_worktree.patch
# place ckpts/deimv2_x_coco.pth + ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
# copy single_stage/configs/* back into configs/ (same relative layout) if the clone lacks them
```

## Recommendation

Commit this patch inside the DEIMv2 clone (or fork DEIMv2 and push it) so it stops living as an
uncommitted working-tree change — a `git checkout`/reclone in that directory would silently drop it
and break every fish config. Until then, `deimv2_worktree.patch` here is the safety copy, and every
`run_manifest.json` re-captures it per run.
