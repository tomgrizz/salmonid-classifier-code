# Checksums for the published model (v2.7)

A checksum (sha256) is a fingerprint of a file's exact contents: change one byte and the fingerprint changes. Compare the fingerprint of your copy with the one below to know whether you hold the same file that was used for the paper.

Computed on the original workstation on 2026-11-10 by reading the files listed. To check a file:

```
python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>
```

## External files (not in git; see `docs/DATA_AVAILABILITY.md`)

| What | File name | Location in `salmonid/config.py` | Bytes | sha256 |
|---|---|---|---:|---|
| Published model weights | `checkpoint0017.pth` | `config.CHECKPOINT_V27` | 819,396,271 | `5cd9bda8a77b01ae07af87e77dcab9b20acffaf86f2e0b3c03dcfc3205e4238e` |
| Starting weights (COCO-pretrained DEIMv2-X detector) | `deimv2_x_coco.pth` | `config.DEIM_INIT_COCO` | 205,523,919 | `b0eaeb2cbfd96e6d98437b1b9cb8ce387582ccf74e0ab467c3cd11426c309665` |
| Backbone weights (DINOv3 ViT-S+/16) | `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` | `config.DEIM_CKPTS / <name>` | 114,924,783 | `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea` |
| Training picture list | `train.json` | `config.COCO_V27 / train.json` | 18,973,783 | `a9a0eed56ab0472bf6a74d7e975e9c53b12ca8892fe470f5649a191f4c794a38` |
| Validation picture list | `val_clean.json` | `config.COCO_V27 / val_clean.json` | 916,513 | `e33e7db164a47048feca52c5691eac1fb662425569a2499c24431510474ef12c` |

The two picture lists carry the same sha256 and byte size in `run_manifest.json`, which was written before training started. That equality was re-checked when this file was generated.

## Committed files in `data/`

| What | File | Bytes | sha256 |
|---|---|---:|---|
| Held-out test set: clips and fish kept out of training | `data/test_set_v6_holdout.json` | 108,248 | `820ff9f9112ccc2dd51f36883c8cc599eb20e7aa89bd2223ae103ed9131e578f` |
| Held-out test set: the species answers | `data/test_set_v6_gt.json` | 74,941 | `d0ec3307554f65b3e7b3d5c7cce806444c56e5e13e851e9a6318290c239d55c4` |
| Published model's predictions on the held-out set | `data/preds_v27_ckpt0017.csv` | 29,713 | `63f25a8567299a782c0291c46b0cb94a17a5cd84bd633ab03a52fba5247553ac` |

These are text files and git stores them with LF line endings; the values above are for the LF form, which is what a clone on Linux or macOS contains (a Windows clone may check them out with CRLF endings, which changes the fingerprint but not the content). The originals on the project data drive, against which the training data were built and the model was scored, were written on Windows with CRLF endings. Converted to LF they are identical to the committed files (re-checked when this file was generated):

| Original on the project data drive | Bytes | sha256 |
|---|---:|---|
| `label_batches/test_set_v6_holdout.json` | 111,549 | `678f44bc6afad8329f7a3d03a1c7fd4f39a2981626b388f45b374538470be1fc` |
| `label_batches/test_set_v6_gt.json` | 77,508 | `e54866e867017fdd471f051ee2f632978d8260e0be64295f7e44e57d077be2ac` |
| `deimv2_x_v27_eval/preds_0017.csv` (committed as `data/preds_v27_ckpt0017.csv`) | 30,080 | `aa0c2df755a7ccffe3930e99e6809d61c6b6f9767b5dbe0d532ffcc5f8582000` |

## The run records in this folder

| File | Bytes as committed | sha256 as committed | sha256 of the original on the workstation |
|---|---:|---|---|
| `run_manifest.json` | 4,514 | `f549cc0747f83677c7f7616f49f46643c135f6547a8a518195684cd6a9425387` | `8c07b3006efad90f1f958b12fdac5b03b5dc7fde7984a57ba472a6c138ed1201` (4,648 bytes, CRLF line endings) |
| `config_snapshot/coco_detection.yml` | 826 | `d5641faa4ae3b214fda299648bf34969d03c9aeb0062cafebd666b019d489fda` | `00da5dc4f1cdb6028a8cf8f040655bb60897a663d13ecacf8ae2dd738f4e818c` (866 bytes, CRLF line endings) |
| `config_snapshot/dataloader.yml` | 1,151 | `e71f4b39251ea9feb60aa4eaf404de1b3ca8f57399af1bc7ca406a10e92647d3` | `5006a2a71ed8a96f1393c0487936c48b26c2066cf28f7e4a07f2f97d3ac1f719` (1,190 bytes, CRLF line endings) |
| `config_snapshot/deimv2.yml` | 3,308 | `718e07aea89d3f8dd6b741ed4ecf71f1f58012ef67975400fbee86bc944d3450` | `c0ca8367887f5f1cf214f5231a2b0d51014e494d8cca441f17f375e990362dfc` (3,451 bytes, CRLF line endings) |
| `config_snapshot/deimv2_dinov3_x_coco.yml` | 2,561 | `182e65e1c7975cdb35ddbd720f686769d23c95d12dd881978c7df7fd9eecf5b3` | `a25a8f3f21f4f6f5318dc30e9aaf9de75396d58a7b0109a78fa64c210e78fa20` (2,654 bytes, CRLF line endings) |
| `config_snapshot/deimv2_fish_x_v27.yml` | 3,933 | `a31fb343ee2151b4a5fb8ca3499903a11240a180d77b28d3c8772a9a2e995a6d` | same |
| `config_snapshot/optimizer.yml` | 522 | `176d885542d44aaa8b92a273c9574bca63ecd6b95a39ea9522b48535bf38b94b` | `fbc24afa676f2070164ded4812a90480b24f15b3b281b965690123d62f0972a5` (557 bytes, CRLF line endings) |
| `config_snapshot/runtime.yml` | 233 | `8339f6a77f21c1ae3f43740b9abecb5e9fb73befde3c4ad5d53e1e23f2610d97` | `042fecf70c24ca408113dc6456106ef1c7b8b6c3f5daa78482b71a6635683abd` (253 bytes, CRLF line endings) |
| `deimv2_worktree.patch` | 2,162 | `c1e9a38f48b237e618b183c7390b5b826bf6119b30b424238c0c2ea421a2c143` | same |

Where the two differ, the only difference is the line endings: the originals were written on Windows (CRLF) and git stores text files with LF. `deimv2_worktree.patch` is stored byte for byte.
