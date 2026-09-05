"""Run a reproducible four-way ablation for underwater fish detection."""

from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "runs" / "underwater-fish-complete"
PRETRAINED = ROOT / "yolov8n.pt"
DATA_ORIGINAL = ROOT / "ultralytics" / "cfg" / "datasets" / "underwater-fish.yaml"
DATA_CLAHE = ROOT / "ultralytics" / "cfg" / "datasets" / "underwater-fish-clahe.yaml"
DATA_ROBUST = ROOT / "ultralytics" / "cfg" / "datasets" / "underwater-fish-robust.yaml"
DATA_ROBUST_CLAHE = ROOT / "ultralytics" / "cfg" / "datasets" / "underwater-fish-robust-clahe.yaml"
CBAM_CFG = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-cbam.yaml"
DFCN_CFG = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-dfcn.yaml"

# Source YOLOv8n layer index -> augmented model layer index.
LAYER_MAPS = {
    "cbam": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 9, 8: 10, 9: 11, 10: 13, 11: 14, 12: 15,
             13: 16, 14: 17, 15: 18, 16: 19, 17: 20, 18: 21, 19: 22, 20: 23, 21: 24, 22: 25},
    "dfcn": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 7, 6: 8, 7: 11, 8: 12, 9: 13, 10: 16, 11: 17, 12: 18,
             13: 19, 14: 20, 15: 21, 16: 22, 17: 23, 18: 24, 19: 25, 20: 26, 21: 27, 22: 28},
}


def remap_pretrained(model: YOLO, variant: str) -> int:
    """Transfer every shape-compatible YOLOv8n parameter across inserted layers."""
    source = YOLO(str(PRETRAINED)).model.state_dict()
    target = model.model.state_dict()
    reverse_map = {target_idx: source_idx for source_idx, target_idx in LAYER_MAPS[variant].items()}
    copied = 0
    for target_key in list(target):
        match = re.match(r"model\.(\d+)\.(.+)", target_key)
        if not match:
            continue
        target_idx, suffix = int(match.group(1)), match.group(2)
        source_idx = reverse_map.get(target_idx)
        if source_idx is None:
            continue
        source_key = f"model.{source_idx}.{suffix}"
        if source_key in source and source[source_key].shape == target[target_key].shape:
            target[target_key] = source[source_key]
            copied += 1
    model.model.load_state_dict(target, strict=True)
    return copied


def metrics_dict(metrics) -> dict[str, float]:
    """Extract stable metric names from Ultralytics metric objects."""
    results = metrics.results_dict
    return {
        "precision": float(results["metrics/precision(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "map50": float(results["metrics/mAP50(B)"]),
        "map50_95": float(results["metrics/mAP50-95(B)"]),
    }


def build_model(spec: dict) -> tuple[YOLO, int]:
    """Create a baseline or augmented detector with the same pretrained initialization."""
    if spec["architecture"] == "baseline":
        return YOLO(str(PRETRAINED)), 0
    model = YOLO(str(spec["architecture"]))
    return model, remap_pretrained(model, spec["name"])


def checkpoint_initialized_model(model: YOLO, variant: str) -> Path:
    """Persist the mapped custom model so Ultralytics' trainer receives it as pretrained weights."""
    checkpoint_path = PROJECT / f"initialized-{variant}.pt"
    network = deepcopy(model.model).half()
    network.pt_path = str(checkpoint_path)
    torch.save({"model": network, "train_args": {"model": str(checkpoint_path), "task": "detect"}}, checkpoint_path)
    return checkpoint_path


def run_one(spec: dict, args: argparse.Namespace) -> dict:
    """Train one ablation member, then evaluate clean and degraded held-out sets."""
    model, copied = build_model(spec)
    if spec["architecture"] != "baseline":
        model = YOLO(str(checkpoint_initialized_model(model, spec["name"])))
    run_name = f"{spec['name']}-{args.tag}"
    train_args = dict(
        data=str(spec["data"]), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device="cpu", workers=0,
        optimizer="AdamW", lr0=args.lr0, lrf=0.01, cos_lr=True, warmup_epochs=1.0, close_mosaic=2,
        seed=args.seed, deterministic=True, pretrained=True, project=str(PROJECT), name=run_name, exist_ok=True,
        plots=False, save=True, val=True, cache=False,
    )
    model.train(**train_args)
    best = PROJECT / run_name / "weights" / "best.pt"
    trained = YOLO(str(best))
    clean_metrics = trained.val(data=str(spec["data"]), split="test", imgsz=args.imgsz, batch=args.batch, device="cpu", workers=0, plots=False, verbose=False)
    robust_data = DATA_ROBUST_CLAHE if spec["uses_clahe"] else DATA_ROBUST
    robust_metrics = trained.val(data=str(robust_data), split="test", imgsz=args.imgsz, batch=args.batch, device="cpu", workers=0, plots=False, verbose=False)
    return {
        "name": spec["name"], "run_name": run_name, "description": spec["description"],
        "parameters": sum(parameter.numel() for parameter in model.model.parameters()), "pretrained_tensors_copied": copied,
        "clean_test": metrics_dict(clean_metrics), "robust_test": metrics_dict(robust_metrics),
        "weights": str(best),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="complete")
    parser.add_argument("--only", choices=("baseline", "clahe", "cbam", "dfcn"))
    args = parser.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "yolo-config"))
    specs = [
        {"name": "baseline", "description": "YOLOv8n", "architecture": "baseline", "data": DATA_ORIGINAL, "uses_clahe": False},
        {"name": "clahe", "description": "YOLOv8n + CLAHE", "architecture": "baseline", "data": DATA_CLAHE, "uses_clahe": True},
        {"name": "cbam", "description": "YOLOv8n + CLAHE + ResidualCBAM", "architecture": CBAM_CFG, "data": DATA_CLAHE, "uses_clahe": True},
        {"name": "dfcn", "description": "YOLOv8n + CLAHE + ResidualCBAM + FGFC", "architecture": DFCN_CFG, "data": DATA_CLAHE, "uses_clahe": True},
    ]
    if args.only:
        specs = [spec for spec in specs if spec["name"] == args.only]
    PROJECT.mkdir(parents=True, exist_ok=True)
    report = []
    for spec in specs:
        result = run_one(spec, args)
        report.append(result)
        (PROJECT / "ablation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
