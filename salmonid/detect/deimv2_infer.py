"""In plain terms: this file loads the trained fish-recognition model and runs it on video
frames. For each frame it returns a box around every fish it finds, together with a score
between 0 and 1 for each possible species. It is the first link in the chain: later steps
follow each fish from frame to frame and add up the counts. Nothing in this file counts fish
or decides a species on its own; it only reports what the model sees in one frame.

DEIMv2 detector wrapper (the DETECT stage).

Pipeline stage: data -> **detect (this module)** -> track -> count -> results -> figures.

Inputs : the model architecture file (``config.INFER_CONFIG``), the trained weights of the
         published model (``config.CHECKPOINT_V27``, v2.7 epoch 17), and RGB video frames
         (height x width x 3, 8-bit; the fishway footage is 1280 x 960 px).
Outputs: per-frame detections held in memory: boxes in the pixel coordinates of the original
         frame, and for each box the six class scores (five species + "Small Salmonid").
         Nothing is written to disk here; ``detect_store.py`` saves the results.
Feeds  : every model-derived number in the manuscript, indirectly. The held-out species
         score (Table 1, Figure 2) and all counts (Tables 2-5, Figures 4-7) start from the
         detections this wrapper produces.
Run    : not run directly. It is used by ``python -m salmonid.detect.detect_store`` and
         ``python -m salmonid.eval.deimv2_bbox_gate``.
Tier   : 2 (needs the external DEIMv2 code, the model weights and a GPU; see
         docs/DEIMV2_PIN.md and docs/DATA_AVAILABILITY.md).

Environment
-----------
Must run in the separate DEIMv2 Python environment (requirements-deim.txt: torch 2.7,
torchvision 0.22), not the base environment. The DEIMv2 code itself is an external
Apache-2.0 repository that is not copied into this package; ``config.DEIM_DIR`` says where
your clone is. Importing this module does NOT need that clone and does not change the
working directory. The clone is only looked for when a detector is actually built (see
``_load_engine``); if it is missing you get one sentence saying what to download.

Facts a reader should know about how frames are handled
-------------------------------------------------------
* Frames are resized from 1280 x 960 to the model's 960 x 960 input. This squeezes the
  picture horizontally (it is not padded to keep its shape). Training did exactly the same,
  so the model sees fish the way it learned them. Boxes are scaled back to the original
  1280 x 960 pixels before they are returned, so all later distances are in real frame pixels.
* The top 40 rows of each frame carry a burned-in date and river name. They are blacked out
  (``mask_header_px=40``) because a model could otherwise "read" the river or season from the
  text instead of looking at the fish. The model was trained with this mask, so inference
  must use the same value.
* Class order is fixed: 0 Chinook, 1 Coho, 2 Atlantic, 3 Rainbow Trout, 4 Brown Trout,
  5 Small Salmonid (juveniles too small to identify; a training-only class).
* The six scores are independent ("sigmoid") scores, one per class. They are not
  probabilities that sum to 1.

Coordinate convention (for maintainers): DEIMv2's own post-processing takes the original
frame size as [[width, height]], width first, and ``predict`` passes it that way. The
``predict_full*`` methods skip that post-processing and scale the model's normalised boxes
by [W, H, W, H] themselves, which is exact for the plain resize used here.
"""
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from salmonid import config as _cfg

# The six classes the detector was trained on. The list is one longer than
# salmonid.config.SPECIES, which holds only the five species the paper reports.
SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout", "Small Salmonid"]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_engine():
    """Make the external DEIMv2 code importable and return its ``YAMLConfig`` class.

    The original research script did this at import time (it added the DEIMv2 clone to the
    module search path and changed the working directory as soon as the file was imported).
    That made every relative path on the command line silently resolve inside the DEIMv2
    clone, and it crashed with a raw traceback on any machine without the clone. Here the same
    two steps happen only when a detector is built, and a missing clone stops with a single
    clear message.
    """
    engine_dir = _cfg.require(
        _cfg.DEIM_DIR / "engine",
        "the DEIMv2 detector source code (the 'engine' folder of your DEIMv2 clone; "
        "see docs/DEIMV2_PIN.md for the exact version)",
        "SALMONID_DEIM_DIR")
    deim_dir = str(engine_dir.parent)
    if deim_dir not in sys.path:
        sys.path.insert(0, deim_dir)
    from engine.core import YAMLConfig  # external dependency, not part of this repository
    return YAMLConfig


@contextlib.contextmanager
def _inside(directory):
    """Temporarily make ``directory`` the working directory, then put the old one back.

    DEIMv2's configuration files name some files relative to the DEIMv2 clone (for example
    the backbone's starting weights), so the model has to be *built* from inside the clone.
    Once it is built nothing depends on the working directory, so we return to where the
    user started. Their relative output paths then mean what they expect.
    """
    previous = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_input(path, what):
    """Turn a config / checkpoint path into an absolute path, or stop with a clear message.

    A relative path is looked up first from where the command was run, then inside the DEIMv2
    clone. The second lookup keeps the original commands working: they named the checkpoint
    relative to the clone, e.g. ``outputs/deimv2_fish_x_v27/checkpoint0017.pth``.
    """
    p = Path(path)
    if not p.is_absolute():
        here = Path.cwd() / p
        p = here if here.exists() else (_cfg.DEIM_DIR / p)
    return str(_cfg.require(p, what, "SALMONID_DEIM_DIR"))


class DEIMv2Detector:
    """The trained fish detector: build it once, then call it on frames.

    Three ways to call it:

    * ``predict_full_batch`` - the one used for every published number. Takes a stack of
      frames and returns, for each frame, the boxes and the full six-score vector per box.
    * ``predict_full`` - the same result for a single frame.
    * ``predict`` - an older path that returns only the single best class per box. Kept for
      the bounding-box quality check (``eval/deimv2_bbox_gate.py``); not used for counts.
    """

    def __init__(self, config: str, ckpt: str, device: str = "cuda", mask_header_px: int = 0,
                 channels_last: bool = False, fp16: bool = False):
        """Build the detector from an architecture file and a weights file.

        Args:
            config: architecture file (``config.INFER_CONFIG``).
            ckpt: trained weights (``config.CHECKPOINT_V27`` for the published model).
            device: where to run it. The published runs used a GPU ("cuda").
            mask_header_px: black out this many rows at the top of each frame before detection.
                The published model was trained with 40, so use 40. 0 switches the mask off.
            channels_last: a memory-layout hint that can speed things up slightly. It changes
                scores only in about the sixth decimal place. Not used for the published runs.
            fp16: run the network in half precision. About twice as fast; scores change in
                about the third decimal place. The season-long count runs used it. The
                held-out species benchmark did not.
        """
        self.mask_header_px = mask_header_px
        self.channels_last = channels_last
        self.fp16 = fp16
        config = _resolve_input(config, "the model architecture file (--config)")
        ckpt = _resolve_input(ckpt, "the trained model weights (--ckpt)")
        YAMLConfig = _load_engine()
        with _inside(_cfg.DEIM_DIR):
            cfg = YAMLConfig(config, resume=ckpt)
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
            checkpoint = torch.load(ckpt, map_location="cpu")
            # A checkpoint holds two copies of the weights. "ema" is a running average over
            # training steps; it is steadier than the raw copy and is the one the DEIMv2
            # authors use for inference, so we do too.
            state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
            cfg.model.load_state_dict(state)
            self.vit = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
            size = cfg.yaml_cfg["eval_spatial_size"]  # [height, width]; 960 x 960 here
            self.size = (int(size[0]), int(size[1]))

            class _M(nn.Module):
                """The network plus DEIMv2's own post-processing, as one callable.

                ``forward`` gives the post-processed result (used by ``predict``); ``.model``
                stays reachable so the ``predict_full*`` methods can read the raw scores.
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
        # Frame preparation, identical to training: resize to 960 x 960 (a plain resize that
        # squeezes the 1280-wide frame; no padding), convert to numbers in 0-1, then subtract
        # the standard colour mean and divide by the standard spread that the backbone expects.
        self.tf = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            (T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD) if self.vit
             else T.Lambda(lambda x: x)),
        ])

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, thresh: float = 0.3):
        """One frame in, best-class detections out (older path; not used for counts).

        Returns ``(boxes, scores, labels)`` for detections whose score is above ``thresh``.
        Boxes are x1, y1, x2, y2 in the pixels of the original frame.
        """
        if self.mask_header_px > 0:
            rgb = rgb.copy(); rgb[:self.mask_header_px, :, :] = 0
        pil = Image.fromarray(rgb)
        W, H = pil.size  # PIL gives (width, height)
        im = self.tf(pil).unsqueeze(0).to(self.device)
        orig = torch.tensor([[W, H]], device=self.device)  # [[width, height]]: width first
        labels, boxes, scores = self.model(im, orig)
        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        keep = scores > thresh
        return boxes[keep], scores[keep], labels[keep].astype(int)

    @torch.no_grad()
    def predict_full(self, rgb: np.ndarray, thresh: float = 0.3):
        """One frame in; boxes and the full six-score vector per box out.

        The model proposes a fixed number of candidate boxes per frame. A candidate is kept
        when its highest class score is above ``thresh`` (0.30 in every published run). Note
        that the highest score is taken over all six classes, so a candidate can be kept on
        the strength of its "Small Salmonid" score alone.

        Returns ``(boxes, scores)``: boxes are x1, y1, x2, y2 in original-frame pixels and
        ``scores`` has one row per box and six columns in the class order given above.
        This single-frame path always runs in full precision.
        """
        if self.mask_header_px > 0:
            rgb = rgb.copy(); rgb[:self.mask_header_px, :, :] = 0
        pil = Image.fromarray(rgb)
        W, H = pil.size
        im = self.tf(pil).unsqueeze(0).to(self.device)
        out = self.model.model(im)  # raw network output: {'pred_logits', 'pred_boxes'}
        logits = out["pred_logits"][0]            # [candidates, classes]
        bcxcywh = out["pred_boxes"][0]            # [candidates, 4]: centre x, centre y, width, height, each 0-1
        probs = logits.sigmoid()                  # independent 0-1 score per class
        maxp = probs.max(dim=-1).values
        keep = maxp > thresh
        probs = probs[keep]
        b = bcxcywh[keep]
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        scale = torch.tensor([W, H, W, H], device=b.device, dtype=xyxy.dtype)
        xyxy = xyxy * scale                       # back to original-frame pixels
        return xyxy.cpu().numpy(), probs.cpu().numpy()

    @torch.no_grad()
    def predict_full_batch(self, rgbs, thresh: float = 0.3):
        """Several frames in one pass; for each frame, boxes and the six-score vector per box.

        This is the method behind every published number. It does the same arithmetic as
        ``predict_full`` but sends a stack of frames through the network at once, which is
        much faster. Processing frames together rather than one at a time changes the results
        only by rounding inside the GPU (about half a pixel in a box edge and 0.0004 in a
        score); the paper's numbers were all produced with this batched method, so that
        rounding is already part of them.

        ``rgbs`` is a list of frames. Returns a list of ``(boxes, scores)`` pairs, one per
        frame, in the same format as ``predict_full``.
        """
        if not rgbs:
            return []
        ims, sizes = [], []
        for rgb in rgbs:
            if self.mask_header_px > 0:
                rgb = rgb.copy(); rgb[:self.mask_header_px, :, :] = 0
            pil = Image.fromarray(rgb)
            sizes.append(pil.size)                       # (width, height)
            ims.append(self.tf(pil))
        im = torch.stack(ims).to(self.device, non_blocking=True)
        if self.channels_last:
            im = im.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model.model(im)                   # raw network output for the whole stack
        probs = out["pred_logits"].float().sigmoid()     # [frames, candidates, classes]; back to full precision first
        b = out["pred_boxes"].float()                    # [frames, candidates, 4], each 0-1
        cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        probs_np = probs.cpu().numpy()                   # one transfer from the GPU for the whole stack
        xyxy_np = xyxy.cpu().numpy()
        results = []
        for i, (W, H) in enumerate(sizes):
            p = probs_np[i]                              # [candidates, classes]
            keep = p.max(axis=-1) > thresh               # keep a candidate if its best class score > 0.30
            box = xyxy_np[i][keep] * np.array([W, H, W, H], dtype=xyxy_np.dtype)
            results.append((box, p[keep]))
        return results
