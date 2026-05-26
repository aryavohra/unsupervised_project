#!/usr/bin/env python3
"""Plot XSA geometry diagnostics from an ``*_xsa_geometry.npz`` artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SKIPPED_LAYERS = {6}
PAIRED_LAYERS = {0, 2, 5, 9}


def _as_mode_slice(values: np.ndarray, mode_index: int, mode_count: int | None) -> np.ndarray:
    """Return a single-mode view for simple artifacts or stacked sweep artifacts."""
    if mode_count is not None and values.ndim >= 2 and values.shape[0] == mode_count:
        return values[mode_index]
    return values


def _layer_mask(layer_count: int) -> np.ndarray:
    return np.array([layer not in SKIPPED_LAYERS for layer in range(layer_count)])


def _filtered_layers(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    layer_count = values.shape[0]
    mask = _layer_mask(layer_count)
    layers = [layer for layer in range(layer_count) if mask[layer]]
    return values[mask], layers


def _load_metric(data: np.lib.npyio.NpzFile, name: str, mode_index: int) -> np.ndarray:
    if name not in data:
        raise KeyError(f"Missing required metric {name!r} in {data.files}")
    mode_count = len(data["xsa_modes"]) if "xsa_modes" in data else None
    return np.asarray(_as_mode_slice(data[name], mode_index, mode_count), dtype=float)


def _model_span_removed(data: np.lib.npyio.NpzFile, mode_index: int) -> np.ndarray:
    """Return the model-span fraction removed or removable by model-space XSA.

    Older/simple artifacts store ``model_span_frac`` directly. Alignment-sweep
    artifacts can store pre/post values, where the removed fraction is the drop.
    """
    if "model_span_frac_pre" in data and "model_span_frac_post" in data:
        pre = _load_metric(data, "model_span_frac_pre", mode_index)
        post = _load_metric(data, "model_span_frac_post", mode_index)
        return pre - post
    return _load_metric(data, "model_span_frac", mode_index)


def _model_span_removed_by_head(data: np.lib.npyio.NpzFile, mode_index: int) -> np.ndarray:
    """Return per-head model-span fraction removed or removable by model-space XSA."""
    if "model_span_increment_pre" in data and "model_span_increment_post" in data:
        pre = _load_metric(data, "model_span_increment_pre", mode_index)
        post = _load_metric(data, "model_span_increment_post", mode_index)
        return pre - post
    return _load_metric(data, "model_span_increment", mode_index)


def _average_heads(values: np.ndarray) -> np.ndarray:
    """Average a layer-by-head metric over heads, preserving a heatmap column."""
    if values.ndim == 1:
        return values.reshape(-1, 1)
    finite = np.isfinite(values)
    totals = np.where(finite, values, 0.0).sum(axis=1, keepdims=True)
    counts = finite.sum(axis=1, keepdims=True)
    return np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)


def _heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    layers: list[int],
    title: str,
    *,
    cmap: str = "Blues",
    annotate: bool = False,
) -> None:
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel("layer")
    ax.set_yticks(range(len(layers)), [f"L{layer}" for layer in layers])

    if values.shape[1] == 1:
        ax.set_xticks([0], ["head avg"])
    else:
        ax.set_xlabel("head")
        ax.set_xticks(range(values.shape[1]), [f"h{head}" for head in range(values.shape[1])])

    for row, layer in enumerate(layers):
        if layer in PAIRED_LAYERS:
            ax.axhline(row - 0.5, color="black", lw=0.8, alpha=0.45)
            ax.axhline(row + 0.5, color="black", lw=0.8, alpha=0.45)

    if annotate:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                val = values[row, col]
                if np.isfinite(val):
                    ax.text(col, row, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save_heatmap(
    values: np.ndarray,
    layers: list[int],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    _heatmap(ax, values, layers, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_value_fraction_line(values: np.ndarray, layers: list[int], output_path: Path) -> None:
    unpaired_rows = [idx for idx, layer in enumerate(layers) if layer not in PAIRED_LAYERS]
    unpaired_layers = [layers[idx] for idx in unpaired_rows]
    unpaired_values = values[unpaired_rows]
    unpaired_index = np.arange(len(unpaired_layers))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(unpaired_index, unpaired_values, marker="o", color="#0571b0")
    ax.set_title("Value-space fraction removed by unpaired attention layer", fontsize=12, weight="bold")
    ax.set_xlabel("unpaired attention layer index")
    ax.set_ylabel("energy fraction")
    ax.set_ylim(0.0, 0.08)
    ax.set_xticks(unpaired_index)
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_xsa_geometry(npz_path: Path, output_dir: Path, mode_index: int = 0) -> tuple[list[Path], dict[str, float]]:
    data = np.load(npz_path)

    cos_y_v, layers = _filtered_layers(_load_metric(data, "cos_y_v", mode_index))
    removed_value, _ = _filtered_layers(_load_metric(data, "removed_frac_value", mode_index))
    model_span_removed, _ = _filtered_layers(_model_span_removed(data, mode_index))
    model_span_removed_by_head, _ = _filtered_layers(_model_span_removed_by_head(data, mode_index))

    cos_y_v_avg = _average_heads(cos_y_v)
    removed_value_avg = _average_heads(removed_value)
    model_span_removed_avg = _average_heads(model_span_removed_by_head)

    stem = npz_path.stem
    outputs = [
        output_dir / f"{stem}_cos_y_v.png",
        output_dir / f"{stem}_value_fraction_removed.png",
        output_dir / f"{stem}_model_span_fraction_removed.png",
        output_dir / f"{stem}_value_fraction_removed_by_layer.png",
    ]
    _save_heatmap(cos_y_v_avg, layers, r"Mean head-space self-alignment: $\cos(y,v)$", outputs[0])
    _save_heatmap(removed_value_avg, layers, "Mean value-space fraction removed", outputs[1])
    _save_heatmap(
        model_span_removed_avg,
        layers,
        "Mean model-span fraction removed",
        outputs[2],
    )
    _save_value_fraction_line(removed_value_avg[:, 0], layers, outputs[3])
    summary = {
        "mean_value_space_fraction_removed_across_heads": float(np.nanmean(removed_value)),
        "mean_model_space_fraction_removed_across_heads": float(np.nanmean(model_span_removed_by_head)),
        "mean_model_space_fraction_removed_by_layer": float(np.nanmean(model_span_removed)),
    }
    return outputs, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="Path to an *_xsa_geometry.npz artifact.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory. Defaults to <npz-stem>_plots next to the input.",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        default=0,
        help="Mode index for stacked alignment-sweep artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output or args.npz.with_name(f"{args.npz.stem}_plots")
    outputs, summary = plot_xsa_geometry(args.npz, output_dir, mode_index=args.mode_index)
    for output in outputs:
        print(output)
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
