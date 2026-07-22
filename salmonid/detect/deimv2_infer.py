"""In plain terms: this file loads the trained fish-recognition model and runs it on each
video frame to find every fish (drawing a box around it) and name its species. It is the first
step of the analysis; later steps follow each fish across the frames and tally the counts. The
species predictions produced here are what the paper's headline 0.966 accuracy figure and all the
multi-year fish counts are built on.

DEIMv2 single-stage detector inference wrapper (the DETECT stage of the pipeline).

Pipeline placement
------------------
data -> **DETECT (this module)** -> track (BoT-SORT) -> eval/count -> results -> figures.
This is the reusable load-and-predict wrapper around the published single-stage model
(DEIMv2-X, v2.6 -- detection + 5-species classification in one network). Corpus count runs and
the held-out evaluation both instantiate ``DEIMv2Detector`` and call one of the ``predict*``
methods per frame; a downstream tracker + net rule turns the per-frame detections into the
manuscript's per-species counts and net-upstream movement.

Inputs  : a model config YAML + checkpoint (``config.INFER_CONFIG`` + ``config.CHECKPOINT_V26``)
          and RGB frames (HxWx3 uint8; our footage is 1280x960, W>H).
Outputs : per-frame detections -- either the postprocessor's top-1 (boxes/scores/labels) or the
          full per-query class distribution (boxes/probs) used for probability-based per-track
          voting.

Environment: MUST run in **.venv_deim** (the isolated DEIMv2 environment: the DEIMv2 engine +
torch 2.7 / torchvision 0.22), NOT the base env. Mirrors DEIMv2/tools/inference/torch_inf.py's
canonical load path (YAMLConfig + ema.module + .deploy()).

Manuscript claim: this wrapper produces the detections behind the published single-stage counter
(held-out test_set_v6 species macro-F1 0.966) and every multi-year corpus count derived from it.

CRITICAL coordinate convention: DEIMv2's postprocessor takes orig_target_sizes as **[[W, H]]**
(width first) and returns xyxy boxes in the ORIGINAL pixel space. Our frames are 1280x960 (W>H),
so passing [H,W] would distort boxes -- the exact y-bug class we guard against. Verified against
torch_inf.py (`orig_size = torch.tensor([[w, h]])`).

Class labels map DIRECTLY 0-4 = Chinook/Coho/Atlantic/Rainbow/Brown (remap_mscoco_category=False,
deimv2_fish_5class categories id 0-4) -- NO RF-DETR-style +1 shift.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from salmonid import config

# The DEIMv2 engine is an external Apache-2.0 repo (docs/DEIMV2_PIN.md), NOT vendored into this
# package. It must be importable, and YAMLConfig resolves its relative config/ckpt paths from the
# DEIMv2 repo root -- so add that repo to sys.path and chdir into it before importing engine.core.
# config.DEIM_DIR resolves to the same absolute location the original hard-coded literal pointed at.
sys.path.insert(0, str(config.DEIM_DIR))
os.chdir(config.DEIM_DIR)  # YAMLConfig + relative ckpt paths resolve from the repo root
from engine.core import YAMLConfig  # noqa: E402  # external dependency (not vendored in this repo)

# 6-class detector vocabulary (5 report species + "Small Salmonid"); the extra class is why this
# list is one longer than salmonid.config.SPECIES, which holds only the 5 reported species.
SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout", "Small Salmonid"]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DEIMv2Detector:
    """Load a trained DEIMv2 single-stage fish detector and run per-frame inference.

    Wraps YAMLConfig-driven model construction, EMA-weight loading, and the deployable
    model+postprocessor into a single callable. Exposes three inference entry points:
    ``predict`` (postprocessor top-1), ``predict_full`` (full per-query class distribution,
    single frame), and ``predict_full_batch`` (batched full-distribution forward for throughput).
    """

    def __init__(self, config: str, ckpt: str, device: str = "cuda", mask_header_px: int = 0,
                 channels_last: bool = False, fp16: bool = False):
        """Build the deployable detector from a config YAML + checkpoint.

        Args:
            config: path to the DEIMv2 model config YAML (the X arch def, e.g. config.INFER_CONFIG).
            ckpt: path to the checkpoint (.pth); EMA weights are preferred when present.
            device: torch device to place the model on ("cuda" by default).
            mask_header_px: black out the top N rows before detection; MUST match training.
            channels_last: use channels_last memory format (5090 speed-up, ~1e-6 differences).
            fp16: autocast the forward for the corpus count run (~1e-3 score perturbation).

        Side effects: loads the checkpoint from disk, constructs and moves the model to ``device``
        in eval mode, and stores the eval-resolution transform pipeline on ``self``.
        """
        # mask_header_px: black out the top N rows before detection (date/river overlay shortcut).
        # MUST match the mask the model was TRAINED with. 0 = off (unmasked legacy models).
        # channels_last: memory-layout hint that speeds the ViT/conv forward on the 5090. Values are
        # identical to NCHW up to ~1e-6 reduction-order rounding -- gate it on the detection-identity
        # check (scripts/_verify_batch_identity.py) before trusting it for an eval-grade run.
        # fp16: autocast the forward (~1.5-2.5x on the ViT). Perturbs scores ~1e-3 -> use for the corpus
        # COUNT run (count-r is scale-robust; matches the two-stage --fp16 precedent), NOT for exact
        # eval-AP/F1 reproduction. Stored probs are cast back to float32.
        self.mask_header_px = mask_header_px
        self.channels_last = channels_last
        self.fp16 = fp16
        cfg = YAMLConfig(config, resume=ckpt)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)
        self.vit = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
        size = cfg.yaml_cfg["eval_spatial_size"]  # [H, W] -> Resize takes (h, w)
        self.size = (int(size[0]), int(size[1]))

        class _M(nn.Module):
            """Fuse the deployable model and postprocessor into one forward pass.

            forward() returns the postprocessor output (labels, boxes, scores); ``.model`` is
            kept accessible so the ``predict_full*`` paths can read the raw detector head directly.
            """
            def __init__(s):
                super().__init__()
                s.model = cfg.model.deploy()
                s.post = cfg.postprocessor.deploy()
            def forward(s, images, orig_target_sizes):
                return s.post(s.model(images), orig_target_sizes)

        self.device = device
        self.model = _M().to(device).eval()
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.tf = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            (T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD) if self.vit
             else T.Lambda(lambda x: x)),
        ])

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, thresh: float = 0.3):
        """rgb: HxWx3 uint8. Returns (boxes_xyxy[N,4] pixel, scores[N], labels[N]) for score>thresh."""
        if self.mask_header_px > 0:
            rgb = rgb.copy(); rgb[:self.mask_header_px, :, :] = 0
        pil = Image.fromarray(rgb)
        W, H = pil.size  # PIL size = (width, height)
        im = self.tf(pil).unsqueeze(0).to(self.device)
        orig = torch.tensor([[W, H]], device=self.device)  # [[W, H]] -- width first!
        labels, boxes, scores = self.model(im, orig)
        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        keep = scores > thresh
        return boxes[keep], scores[keep], labels[keep].astype(int)

    @torch.no_grad()
    def predict_full(self, rgb: np.ndarray, thresh: float = 0.3):
        """Like predict(), but returns the FULL per-detection class distribution instead of the
        postprocessor's top-1. Bypasses the flatten-topk PostProcessor and reads the raw model
        head: per query -> sigmoid over all classes. Keeps queries whose MAX class score > thresh.
        Returns (boxes_xyxy[N,4] pixel, probs[N,C] sigmoid). Used by the full-distribution re-store
        so per-track voting uses real Coho/Chinook probabilities, not one-hot argmax counts."""
        if self.mask_header_px > 0:
            rgb = rgb.copy(); rgb[:self.mask_header_px, :, :] = 0
        pil = Image.fromarray(rgb)
        W, H = pil.size
        im = self.tf(pil).unsqueeze(0).to(self.device)
        out = self.model.model(im)  # raw detector head: {'pred_logits','pred_boxes'}
        logits = out["pred_logits"][0]            # [Q, C]
        bcxcywh = out["pred_boxes"][0]            # [Q, 4] normalized cxcywh
        probs = logits.sigmoid()                  # [Q, C]
        maxp = probs.max(dim=-1).values
        keep = maxp > thresh
        probs = probs[keep]
        b = bcxcywh[keep]
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        scale = torch.tensor([W, H, W, H], device=b.device, dtype=xyxy.dtype)
        xyxy = xyxy * scale
        return xyxy.cpu().numpy(), probs.cpu().numpy()

    @torch.no_grad()
    def predict_full_batch(self, rgbs, thresh: float = 0.3):
        """Batched predict_full: ONE forward over a stack of frames + ONE host transfer, so the 5090
        stays saturated (vs one frame per launch + a per-frame .cpu() sync). Numerically identical to
        calling predict_full() on each frame: the per-image transform (self.tf) is applied per frame
        then stacked, the raw-head forward is per-sample independent (DINOv3 LayerNorm is per-token,
        eval-mode BN uses running stats), and the sigmoid/threshold/cxcywh->xyxy math is unchanged.

        rgbs: list[HxWx3 uint8]. Returns list[(boxes_xyxy[N,4] pixel, probs[N,C] sigmoid)], one per frame.
        """
        if not rgbs:
            return []
        ims, sizes = [], []
        for rgb in rgbs:
            if self.mask_header_px > 0:
                rgb = rgb.copy(); rgb[:self.mask_header_px, :, :] = 0
            pil = Image.fromarray(rgb)
            sizes.append(pil.size)                       # (W, H)
            ims.append(self.tf(pil))
        im = torch.stack(ims).to(self.device, non_blocking=True)
        if self.channels_last:
            im = im.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model.model(im)                   # raw head, batched: {'pred_logits','pred_boxes'}
        probs = out["pred_logits"].float().sigmoid()     # [B, Q, C] (cast fp16->fp32 before sigmoid)
        b = out["pred_boxes"].float()                    # [B, Q, 4] normalized cxcywh
        cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)  # [B, Q, 4] norm
        probs_np = probs.cpu().numpy()                   # single transfer for the whole batch
        xyxy_np = xyxy.cpu().numpy()
        results = []
        for i, (W, H) in enumerate(sizes):
            p = probs_np[i]                              # [Q, C]
            keep = p.max(axis=-1) > thresh
            box = xyxy_np[i][keep] * np.array([W, H, W, H], dtype=xyxy_np.dtype)
            results.append((box, p[keep]))
        return results
