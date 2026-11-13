# The external detector code (DEIMv2) and how to obtain the exact version

The model is a DEIMv2 detector. Training and detection run the DEIMv2 code published by its
authors (`Intellindust-AI-Lab/DEIMv2` on GitHub), which is **not** copied into this repository.
To reproduce training or detection you need that code in exactly the state below.

## The pinned state

| | |
|---|---|
| Commit | `0fff8d4dcdc272e6cf2d84be31399db471357941` (also in `DEIMV2_SHA.txt`), dated 2026-03-24 |
| Licence at that commit | Apache-2.0. The upstream project changed its licence on 2026-08-24; the terms of the pinned commit are the ones that apply to it. See `SOFTWARE_CITATIONS.md`. |
| Required patch | `deimv2_worktree.patch` (in this folder), applied on top of the commit |
| Configuration files | `../configs/`, copied into the DEIMv2 `configs/` folder |
| Starting weights | `ckpts/deimv2_x_coco.pth` and `ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`; checksums in `../provenance/v2.7/CHECKSUMS.md` |

The patch is required, not optional. It makes two changes to one file
(`engine/data/transforms/_transforms.py`):

1. It registers a `ColorJitter` transform. The fish configuration uses it in place of
   DEIMv2's default colour augmentation, which shuffles the colour channels and so destroys
   the colour cue that helps separate silvery Coho Salmon, Rainbow Trout and Chinook Salmon.
   Without the patch the configuration fails to load.
2. It follows a method rename (`_transform` to `transform`) in torchvision 0.22, the version
   used here.

The patch file here is identical to the one recorded with the published training run
(`../provenance/v2.7/deimv2_worktree.patch`), apart from line endings.

## Setting it up

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIMv2
cd DEIMv2
git checkout 0fff8d4dcdc272e6cf2d84be31399db471357941
git apply /path/to/salmonid-classifier-code/docs/deimv2_worktree.patch
cp -r /path/to/salmonid-classifier-code/configs/* configs/
# place the two starting-weight files in ckpts/
```

Then point `SALMONID_DEIM_DIR` at this folder (see `../.env.example`). The training launcher
(`python -m salmonid.train.train_v27 --dry-run`) checks the commit, the patch and the weight
files and says which, if any, is not as expected.

The configuration files that matter for the published model are
`configs/deimv2/deimv2_fish_x_v27.yml` and the files it includes; `../configs/README.md`
describes the rest.
