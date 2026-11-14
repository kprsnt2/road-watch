"""ONNX Runtime wrapper for YOLOv8 / YOLO11 detection models.

Works with any model exported by Ultralytics with ``format='onnx'`` (output
shape [1, 4 + num_classes, N]). The same .onnx file runs on Android through
ONNX Runtime Mobile, so desktop results carry over to the app.

Class names come from the model's metadata, and can be remapped (e.g. COCO's
"motorcycle" is used as-is; a helmet model's "Without Helmet" -> "no_helmet").
"""

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .rules import Box


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain NumPy non-maximum suppression. boxes: (N, 4) x1y1x2y2."""
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


class YoloOnnx:
    def __init__(
        self,
        model_path: Path,
        keep: set[str] | None = None,
        rename: dict[str, str] | None = None,
        conf: float = 0.35,
        iou: float = 0.5,
    ):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        rename = rename or {}
        self.names = {int(k): rename.get(v, v) for k, v in names.items()}
        self.keep = keep
        self.conf, self.iou = conf, iou

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = self.size / max(h, w)
        nh, nw = round(h * r), round(w * r)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def __call__(self, bgr: np.ndarray) -> list[Box]:
        canvas, r, dx, dy = self._letterbox(bgr)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0]  # (4 + nc, N)
        out = out.T
        scores_all = out[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        mask = scores >= self.conf
        if self.keep is not None:
            wanted = [i for i, n in self.names.items() if n in self.keep]
            mask &= np.isin(cls, wanted)
        out, cls, scores = out[mask], cls[mask], scores[mask]
        if not len(out):
            return []
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dx) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dy) / r
        H, W = bgr.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)

        boxes: list[Box] = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            for k in _nms(xyxy[idx], scores[idx], self.iou):
                i = idx[k]
                boxes.append(Box(*map(float, xyxy[i]), self.names.get(int(c), str(c)), float(scores[i])))
        return boxes
