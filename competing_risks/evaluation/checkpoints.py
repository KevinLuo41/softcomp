"""Checkpoint helpers for models, metrics, and cached predictions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def get_output_dir(experiment: str, root: str | Path = "outputs") -> Path:
    path = Path(root) / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_model_checkpoint(model: torch.nn.Module, path: str | Path, metadata: Optional[Dict[str, Any]] = None):
    p = ensure_parent(path)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, p)
    return p


def load_model_checkpoint(model: torch.nn.Module, path: str | Path, map_location: Optional[str] = None):
    checkpoint = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint.get("metadata", {})


def save_model(model: torch.nn.Module, name: str, output_dir: str | Path):
    return save_model_checkpoint(model, Path(output_dir) / "models" / f"{name}.pt")


def load_model(model: torch.nn.Module, name: str, output_dir: str | Path, map_location: Optional[str] = None):
    return load_model_checkpoint(model, Path(output_dir) / "models" / f"{name}.pt", map_location)


def save_pickle(obj: Any, path: str | Path):
    p = ensure_parent(path)
    with p.open("wb") as fh:
        pickle.dump(obj, fh)
    return p


def load_pickle(path: str | Path):
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def save_eval_cache(name: str, output_dir: str | Path, cif_pred: Any, metrics: Dict[str, Any]):
    return save_pickle(
        {"cif_pred": cif_pred, "metrics": metrics},
        Path(output_dir) / "eval_cache" / f"{name}.pt",
    )


def load_eval_cache(name: str, output_dir: str | Path):
    return load_pickle(Path(output_dir) / "eval_cache" / f"{name}.pt")


def save_json(obj: Dict[str, Any], path: str | Path):
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    return p


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_results_txt(results: Dict[str, Any], path: str | Path):
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as fh:
        for key in sorted(results):
            fh.write(f"{key}: {results[key]}\n")
    return p


def save_npz(path: str | Path, **arrays: Any):
    p = ensure_parent(path)
    np.savez_compressed(p, **arrays)
    return p


def load_npz(path: str | Path):
    return np.load(path, allow_pickle=True)
