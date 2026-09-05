"""Run SFS-Conv-inspired underwater fish detection experiments reproducibly."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "yolo-config"))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


PROJECT = ROOT / "runs" / "sfs-inspired-underwater-fish"
PRETRAINED = ROOT / "yolov8n.pt"
DATA_ORIGINAL = ROOT / "ultralytics" / "cfg" / "datasets" / "underwater-fish.yaml"
DATA_CLAHE = ROOT / "ultralytics" / "cfg" / "datasets" / "underwater-fish-clahe.yaml"
DATA_ROBUST = ROOT / "ultralytics" / "cfg" / "datasets" / "underwater-fish-robust.yaml"
DATA_ROBUST_CLAHE = ROOT / "ultralytics" / "cfg" / "datasets" / "underwater-fish-robust-clahe.yaml"
SFS_NECK_CFG = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-sfsneck.yaml"
DFCN_SFS_CFG = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-dfcn-sfs.yaml"

# Source YOLOv8n layer index -> augmented model layer index. New SFS layers are deliberately unmapped.
LAYER_MAPS = {
    "sfs_neck_p3": {**{i: i for i in range(16)}, **{i: i + 1 for i in range(16, 23)}},
    "dfcn_sfs": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 7, 6: 8, 7: 11, 8: 12, 9: 13, 10: 16, 11: 17, 12: 18,
                 13: 19, 14: 20, 15: 21, 16: 22, 17: 23, 18: 24, 19: 25, 20: 26, 21: 27, 22: 28},
}


def remap_pretrained(model: YOLO, variant: str) -> int:
    """Copy source tensors only where both the remapped name and shape match."""
    source = YOLO(str(PRETRAINED)).model.state_dict()
    target = model.model.state_dict()
    reverse_map = {target_index: source_index for source_index, target_index in LAYER_MAPS[variant].items()}
    copied = 0
    for target_key in list(target):
        match = re.match(r"model\.(\d+)\.(.+)", target_key)
        if not match:
            continue
        source_index = reverse_map.get(int(match.group(1)))
        source_key = f"model.{source_index}.{match.group(2)}" if source_index is not None else None
        if source_key in source and source[source_key].shape == target[target_key].shape:
            target[target_key] = source[source_key]
            copied += 1
    model.model.load_state_dict(target, strict=True)
    return copied


def checkpoint_initialized_model(model: YOLO, variant: str) -> Path:
    """Persist mapped weights so the trainer starts from the intended initialization."""
    checkpoint_path = PROJECT / f"initialized-{variant}.pt"
    network = deepcopy(model.model).half()
    network.pt_path = str(checkpoint_path)
    torch.save({"model": network, "train_args": {"model": str(checkpoint_path), "task": "detect"}}, checkpoint_path)
    return checkpoint_path


def metric_values(metrics) -> dict[str, float]:
    results = metrics.results_dict
    return {
        "precision": float(results["metrics/precision(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "map50": float(results["metrics/mAP50(B)"]),
        "map50_95": float(results["metrics/mAP50-95(B)"]),
    }


def cpu_forward_ms(model: torch.nn.Module, imgsz: int) -> float:
    """Measure model-forward latency only; preprocessing and NMS are intentionally excluded."""
    model = model.float().eval()
    image = torch.zeros(1, 3, imgsz, imgsz)
    with torch.inference_mode():
        for _ in range(5):
            model(image)
        start = time.perf_counter()
        for _ in range(20):
            model(image)
    return (time.perf_counter() - start) * 1000.0 / 20.0


def run_one(spec: dict, args: argparse.Namespace) -> dict:
    """Train one SFS-inspired model and evaluate clean and derived degraded test sets."""
    model = YOLO(str(spec["architecture"]))
    copied = remap_pretrained(model, spec["name"])
    initialized = checkpoint_initialized_model(model, spec["name"])
    model = YOLO(str(initialized))
    run_name = f"{spec['name']}-{args.tag}"
    model.train(
        data=str(spec["data"]), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device="cpu", workers=0,
        optimizer="AdamW", lr0=args.lr0, lrf=0.01, cos_lr=True, warmup_epochs=1.0, close_mosaic=2,
        seed=args.seed, deterministic=True, pretrained=True, project=str(PROJECT), name=run_name, exist_ok=True,
        plots=True, save=True, val=True, cache=False,
    )
    best = PROJECT / run_name / "weights" / "best.pt"
    trained = YOLO(str(best))
    # Count the saved single-class detector after the data config overrides nc=1.
    parameters = sum(parameter.numel() for parameter in trained.model.parameters())
    flops = float(get_flops(trained.model, imgsz=args.imgsz))
    clean = trained.val(data=str(spec["data"]), split="test", imgsz=args.imgsz, batch=args.batch, device="cpu", workers=0, plots=True, verbose=False)
    robust_data = DATA_ROBUST_CLAHE if spec["uses_clahe"] else DATA_ROBUST
    robust = trained.val(data=str(robust_data), split="test", imgsz=args.imgsz, batch=args.batch, device="cpu", workers=0, plots=False, verbose=False)
    return {
        "name": spec["name"], "description": spec["description"], "run_name": run_name,
        "parameters": parameters, "gflops_at_imgsz": flops, "cpu_forward_ms": cpu_forward_ms(trained.model, args.imgsz),
        "pretrained_tensors_copied": copied, "clean_test": metric_values(clean), "robust_test": metric_values(robust),
        "weights": str(best), "clean_visuals": str(PROJECT / run_name),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="sfs-inspired-3e")
    parser.add_argument("--only", choices=("sfs_neck_p3", "dfcn_sfs"))
    args = parser.parse_args()
    specs = [
        {"name": "sfs_neck_p3", "description": "YOLOv8n + SFS-inspired block at fused Neck P3", "architecture": SFS_NECK_CFG,
         "data": DATA_ORIGINAL, "uses_clahe": False},
        {"name": "dfcn_sfs", "description": "CLAHE + Residual CBAM + SFS-inspired blocks replacing FGFC", "architecture": DFCN_SFS_CFG,
         "data": DATA_CLAHE, "uses_clahe": True},
    ]
    if args.only:
        specs = [spec for spec in specs if spec["name"] == args.only]
    PROJECT.mkdir(parents=True, exist_ok=True)
    report = []
    for spec in specs:
        result = run_one(spec, args)
        report.append(result)
        (PROJECT / "sfs_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
