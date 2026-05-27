#!/usr/bin/env python3
"""Plot XSA geometry diagnostics from an ``*_xsa_geometry.npz`` artifact."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


SKIPPED_LAYERS = {6}
PAIRED_LAYERS = {0, 2, 5, 9}
SPACE_SPECS = {
    "value": ("removed_frac_value", "Value space"),
    "model": ("model_span_increment", "Model space (per-head contribution; 6 heads)"),
    "residual": ("resid_frac", "Residual space"),
}
SERIES_COLORS = {
    "XSA off": "#d95f02",
    "XSA on": "#0571b0",
}
DENSEVAL_LOG_DIR = Path("novita_denseval_20260526_logs")
DENSEVAL_GATE_OFF_FIXEDCODE = (
    DENSEVAL_LOG_DIR / "record-xsa-gate-off-1400-denseval-stop328-fixedcode-20260526.txt"
)
DENSEVAL_GATE_ON = DENSEVAL_LOG_DIR / "record-xsa-gate-on-1400-denseval-stop328-20260526.txt"
VAL_LOSS_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+"
    r"val_loss:(?P<val_loss>[0-9.]+)\s+"
    r"train_time:(?P<train_time_ms>\d+)ms\s+"
    r"step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)


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


def _unpaired_rows(layers: list[int]) -> list[int]:
    return [idx for idx, layer in enumerate(layers) if layer not in PAIRED_LAYERS]


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


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _value_alignment_by_unpaired_layer(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[int]]:
    removed_value, layers = _filtered_layers(_load_metric(data, "removed_frac_value", 0))
    unpaired_rows = _unpaired_rows(layers)
    unpaired_layers = [layers[idx] for idx in unpaired_rows]
    return _average_heads(removed_value[unpaired_rows])[:, 0], unpaired_layers


def _value_alignment_unpaired_mean(data: np.lib.npyio.NpzFile) -> float:
    removed_value, layers = _filtered_layers(_load_metric(data, "removed_frac_value", 0))
    unpaired_rows = _unpaired_rows(layers)
    values = removed_value[unpaired_rows]
    if "head_count" in data:
        layer_weights, _ = _filtered_layers(np.asarray(data["head_count"], dtype=float))
        weights = np.broadcast_to(layer_weights[unpaired_rows, None], values.shape)
    else:
        weights = np.ones_like(values)
    return _weighted_nanmean(values, weights)


def _series_label(npz_path: Path, data: np.lib.npyio.NpzFile) -> str:
    mode = str(data["xsa_mode"]) if "xsa_mode" in data else ""
    if mode == "none" or npz_path.name.startswith("no-xsa"):
        return "XSA off"
    if mode == "record" or npz_path.name.startswith("record-xsa"):
        return "XSA on"
    return mode or npz_path.stem.split("_val_step", 1)[0]


def _space_alignment_by_unpaired_layer(
    data: np.lib.npyio.NpzFile,
    space: str,
) -> tuple[np.ndarray, list[int]]:
    metric_name, _ = SPACE_SPECS[space]
    values, layers = _filtered_layers(_load_metric(data, metric_name, 0))
    unpaired_rows = _unpaired_rows(layers)
    unpaired_layers = [layers[idx] for idx in unpaired_rows]
    if values.ndim == 1:
        return values[unpaired_rows], unpaired_layers
    return _average_heads(values[unpaired_rows])[:, 0], unpaired_layers


def _space_alignment_unpaired_mean(data: np.lib.npyio.NpzFile, space: str) -> float:
    metric_name, _ = SPACE_SPECS[space]
    values, layers = _filtered_layers(_load_metric(data, metric_name, 0))
    unpaired_rows = _unpaired_rows(layers)
    values = values[unpaired_rows]
    if "head_count" in data and values.ndim == 2:
        layer_weights, _ = _filtered_layers(np.asarray(data["head_count"], dtype=float))
        weights = np.broadcast_to(layer_weights[unpaired_rows, None], values.shape)
    elif "token_count" in data and values.ndim == 1:
        layer_weights, _ = _filtered_layers(np.asarray(data["token_count"], dtype=float))
        weights = layer_weights[unpaired_rows]
    else:
        weights = np.ones_like(values)
    return _weighted_nanmean(values, weights)


def _validation_step(npz_path: Path, data: np.lib.npyio.NpzFile) -> int:
    if "validation_step" in data:
        return int(data["validation_step"])
    match = re.search(r"_val_step(\d+)_xsa_alignment$", npz_path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer validation step from {npz_path}")


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
    unpaired_rows = _unpaired_rows(layers)
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


def _save_value_alignment_over_steps(
    steps: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    *,
    color: str = "#0571b0",
    title: str = "Mean value-space self-alignment over training",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(steps, values, marker="o", color=color, lw=2)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("training step")
    ax.set_ylabel("aligned energy fraction")
    finite = np.isfinite(values)
    if np.any(finite):
        ymax = max(0.08, float(np.nanmax(values[finite])) * 1.15)
        ax.set_ylim(0.0, ymax)
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_value_alignment_by_layer(
    values: np.ndarray,
    layers: list[int],
    output_path: Path,
    *,
    step: int,
    color: str = "#0571b0",
    title: str | None = None,
) -> None:
    positions = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(positions, values, marker="o", color=color, lw=2)
    ax.set_title(title or f"Mean value-space self-alignment by layer at step {step}", fontsize=12, weight="bold")
    ax.set_xlabel("non-paired attention layer")
    ax.set_ylabel("aligned energy fraction")
    ax.set_xticks(positions, [f"L{layer}" for layer in layers])
    finite = np.isfinite(values)
    if np.any(finite):
        ymax = max(0.08, float(np.nanmax(values[finite])) * 1.15)
        ax.set_ylim(0.0, ymax)
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_space_alignment_over_steps(
    series: dict[str, dict[str, dict[str, np.ndarray]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (space, (_, title)) in zip(axes, SPACE_SPECS.items()):
        for label in sorted(series, key=lambda name: (name != "XSA off", name)):
            steps = series[label][space]["steps"]
            values = series[label][space]["means"]
            ax.plot(
                steps,
                values,
                marker="o",
                lw=2,
                color=SERIES_COLORS.get(label, None),
                label=label,
            )
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("training step")
        ax.set_ylabel("aligned energy fraction")
        finite_values = [
            series[label][space]["means"]
            for label in series
            if np.any(np.isfinite(series[label][space]["means"]))
        ]
        if finite_values:
            ymax = max(0.08, float(np.nanmax(np.concatenate(finite_values))) * 1.15)
            ax.set_ylim(0.0, ymax)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Mean self-alignment over training", fontsize=14, weight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_space_alignment_by_layer(
    series: dict[str, dict[str, dict[str, np.ndarray]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    final_steps = []
    for ax, (space, (_, title)) in zip(axes, SPACE_SPECS.items()):
        finite_values = []
        for label in sorted(series, key=lambda name: (name != "XSA off", name)):
            layers = series[label][space]["layers"]
            values = series[label][space]["by_layer"]
            final_steps.append(int(series[label][space]["final_step"]))
            positions = np.arange(len(layers))
            ax.plot(
                positions,
                values,
                marker="o",
                lw=2,
                color=SERIES_COLORS.get(label, None),
                label=label,
            )
            ax.set_xticks(positions, [f"L{layer}" for layer in layers])
            if np.any(np.isfinite(values)):
                finite_values.append(values)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("non-paired attention layer")
        ax.set_ylabel("aligned energy fraction")
        if finite_values:
            ymax = max(0.08, float(np.nanmax(np.concatenate(finite_values))) * 1.15)
            ax.set_ylim(0.0, ymax)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    final_step_label = max(final_steps) if final_steps else "final"
    fig.suptitle(f"Mean self-alignment by layer at step {final_step_label}", fontsize=14, weight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_validation_value_alignment(
    npz_paths: list[Path],
    output_dir: Path,
    stem: str = "validation_value_alignment",
) -> tuple[list[Path], dict[str, float]]:
    if not npz_paths:
        raise ValueError("No validation alignment artifacts were provided")

    records_by_label: dict[str, list[tuple[int, Path]]] = {}
    for npz_path in npz_paths:
        with np.load(npz_path) as data:
            step = _validation_step(npz_path, data)
            label = _series_label(npz_path, data)
        records_by_label.setdefault(label, []).append((step, npz_path))

    series: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for label, records in records_by_label.items():
        records.sort(key=lambda item: item[0])
        steps = np.array([step for step, _ in records], dtype=int)
        series[label] = {}
        for space in SPACE_SPECS:
            means = []
            for _, npz_path in records:
                with np.load(npz_path) as data:
                    means.append(_space_alignment_unpaired_mean(data, space))
            final_step, final_path = records[-1]
            with np.load(final_path) as data:
                by_layer, layers = _space_alignment_by_unpaired_layer(data, space)
            series[label][space] = {
                "steps": steps,
                "means": np.array(means, dtype=float),
                "by_layer": by_layer,
                "layers": np.array(layers, dtype=int),
                "final_step": np.array(final_step, dtype=int),
            }

    outputs = [
        output_dir / f"{stem}_over_steps.png",
        output_dir / f"{stem}_by_unpaired_layer.png",
    ]
    _save_space_alignment_over_steps(series, outputs[0])
    _save_space_alignment_by_layer(series, outputs[1])
    if "XSA off" in series:
        xsa_off_value = series["XSA off"]["value"]
        focused_outputs = [
            output_dir / f"{stem}_xsa_off_value_over_steps.png",
            output_dir / f"{stem}_xsa_off_value_by_unpaired_layer.png",
        ]
        _save_value_alignment_over_steps(
            xsa_off_value["steps"],
            xsa_off_value["means"],
            focused_outputs[0],
            color=SERIES_COLORS["XSA off"],
            title="Baseline value-space self-alignment over training",
        )
        _save_value_alignment_by_layer(
            xsa_off_value["by_layer"],
            [int(layer) for layer in xsa_off_value["layers"]],
            focused_outputs[1],
            step=int(xsa_off_value["final_step"]),
            color=SERIES_COLORS["XSA off"],
            title=f"Baseline value-space self-alignment by layer at step {int(xsa_off_value['final_step'])}",
        )
        outputs.extend(focused_outputs)

    summary = {}
    for label, label_series in series.items():
        label_key = label.lower().replace(" ", "_")
        for space, space_series in label_series.items():
            means = space_series["means"]
            summary[f"{label_key}_{space}_first_step"] = float(space_series["steps"][0])
            summary[f"{label_key}_{space}_last_step"] = float(space_series["steps"][-1])
            summary[f"{label_key}_{space}_first_alignment"] = float(means[0])
            summary[f"{label_key}_{space}_last_alignment"] = float(means[-1])
            summary[f"{label_key}_{space}_peak_alignment"] = float(np.nanmax(means))
    return outputs, summary

def _parse_val_loss_log(log_path: Path) -> np.ndarray:
    rows = []
    for match in VAL_LOSS_RE.finditer(log_path.read_text(errors="ignore")):
        rows.append(
            (
                int(match.group("step")),
                int(match.group("total")),
                float(match.group("val_loss")),
                int(match.group("train_time_ms")) / 1000.0,
                float(match.group("step_avg_ms")),
            )
        )
    if not rows:
        raise ValueError(f"No validation loss rows found in {log_path}")
    return np.array(
        rows,
        dtype=[
            ("step", "i4"),
            ("total", "i4"),
            ("val_loss", "f8"),
            ("train_time_s", "f8"),
            ("step_avg_ms", "f8"),
        ],
    )


def _save_denseval_loss_over_time(
    gate_off_log: Path,
    gate_on_log: Path,
    output_path: Path,
) -> Path:
    gate_off = _parse_val_loss_log(gate_off_log)
    gate_on = _parse_val_loss_log(gate_on_log)
    gate_off = gate_off[gate_off["step"] > 0]
    gate_on = gate_on[gate_on["step"] > 0]

    gate_off_min = gate_off["train_time_s"] / 60.0
    gate_on_min = gate_on["train_time_s"] / 60.0
    late_step = 1370
    gate_off_late = gate_off[gate_off["step"] >= late_step]
    gate_on_late = gate_on[gate_on["step"] >= late_step]
    gate_off_late_min = gate_off_late["train_time_s"] / 60.0
    gate_on_late_min = gate_on_late["train_time_s"] / 60.0

    fig, (ax_full, ax_late) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    ax_full.plot(
        gate_off_min,
        gate_off["val_loss"],
        marker="o",
        markersize=4,
        color="#0571b0",
        label="XSA gate off",
    )
    ax_full.plot(
        gate_on_min,
        gate_on["val_loss"],
        marker="o",
        markersize=4,
        color="#ca0020",
        label="XSA gate on",
    )
    ax_full.axhline(3.28, color="black", lw=1.0, ls="--", alpha=0.45)
    ax_full.set_title("XSA with/without Sparse Attention Gate Ablation", fontsize=12, weight="bold")
    ax_full.set_xlabel("training time (min, log scale)")
    ax_full.set_ylabel("validation loss")
    ax_full.set_xscale("log")
    log_ticks = [0.5, 1.0, 2.0, 5.0, 10.0]
    ax_full.xaxis.set_major_locator(FixedLocator(log_ticks))
    ax_full.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax_full.xaxis.set_minor_formatter(NullFormatter())
    ax_full.grid(True, alpha=0.25)
    ax_full.grid(True, which="minor", alpha=0.12)
    ax_full.legend(frameon=False)

    ax_late.plot(
        gate_off_late_min,
        gate_off_late["val_loss"],
        marker="o",
        markersize=4,
        color="#0571b0",
        label="XSA gate off",
    )
    ax_late.plot(
        gate_on_late_min,
        gate_on_late["val_loss"],
        marker="o",
        markersize=4,
        color="#ca0020",
        label="XSA gate on",
    )
    ax_late.axhline(3.28, color="black", lw=1.0, ls="--", alpha=0.45)
    ax_late.set_title(f"Late window, step {late_step}+", fontsize=12, weight="bold")
    ax_late.set_xlabel("training time (min)")
    ax_late.set_ylabel("validation loss")
    late_losses = np.concatenate([gate_off_late["val_loss"], gate_on_late["val_loss"]])
    ax_late.set_ylim(late_losses.min() - 0.00035, late_losses.max() + 0.00035)
    ax_late.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


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
    summary_rows = _unpaired_rows(layers)
    summary = {
        "mean_value_space_fraction_removed_unpaired_layers_across_heads": float(
            np.nanmean(removed_value[summary_rows])
        ),
        "mean_model_space_fraction_removed_unpaired_layers_across_heads": float(
            np.nanmean(model_span_removed_by_head[summary_rows])
        ),
        "mean_model_space_fraction_removed_unpaired_layers_by_layer": float(
            np.nanmean(model_span_removed[summary_rows])
        ),
    }
    return outputs, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, nargs="+", help="Path(s) to XSA geometry or validation-alignment npz artifacts.")
    parser.add_argument("npz", type=Path, nargs="?", help="Path to an *_xsa_geometry.npz artifact.")
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
    parser.add_argument(
        "--validation-series",
        action="store_true",
        help="Plot validation-step value-space self-alignment over time and by final non-paired layer.",
    )
    parser.add_argument(
        "--stem",
        default="validation_value_alignment",
        help="Output filename stem for --validation-series plots.",
    )
    parser.add_argument(
        "--plot-denseval",
        action="store_true",
        help="Plot dense validation loss over time for the Novita gate-off/gate-on logs.",
    )
    parser.add_argument(
        "--denseval-gate-off-log",
        type=Path,
        default=DENSEVAL_GATE_OFF_FIXEDCODE,
        help="Denseval log for the XSA gate-off fixedcode run.",
    )
    parser.add_argument(
        "--denseval-gate-on-log",
        type=Path,
        default=DENSEVAL_GATE_ON,
        help="Denseval log for the XSA gate-on run.",
    )
    parser.add_argument(
        "--denseval-output",
        type=Path,
        default=DENSEVAL_LOG_DIR / "xsa_gate_denseval_loss_over_time.png",
        help="Output path for the denseval loss-over-time plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_series:
        first_npz = args.npz[0]
        output_dir = args.output or first_npz.with_name(f"{args.stem}_plots")
        outputs, summary = plot_validation_value_alignment(args.npz, output_dir, stem=args.stem)
    else:
        if len(args.npz) != 1:
            raise ValueError("Pass exactly one npz unless --validation-series is set")
        output_dir = args.output or args.npz[0].with_name(f"{args.npz[0].stem}_plots")
        outputs, summary = plot_xsa_geometry(args.npz[0], output_dir, mode_index=args.mode_index)
    for output in outputs:
        print(output)
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")
    if args.plot_denseval:
        print(_save_denseval_loss_over_time(args.denseval_gate_off_log, args.denseval_gate_on_log, args.denseval_output))

    if args.npz is not None:
        output_dir = args.output or args.npz.with_name(f"{args.npz.stem}_plots")
        outputs, summary = plot_xsa_geometry(args.npz, output_dir, mode_index=args.mode_index)
        for output in outputs:
            print(output)
        print("summary:")
        for key, value in summary.items():
            print(f"  {key}: {value:.6f}")
    elif not args.plot_denseval:
        raise SystemExit("Provide an NPZ artifact or pass --plot-denseval.")


if __name__ == "__main__":
    main()
