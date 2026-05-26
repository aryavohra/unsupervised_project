import os
import html

import numpy as np


def _hex_color(value: float, vmin: float, vmax: float, *, diverging: bool = False):
    if not np.isfinite(value):
        return "#d9d9d9"
    if diverging:
        lim = max(abs(vmin), abs(vmax), 1e-12)
        t = max(-1.0, min(1.0, value / lim))
        if t < 0:
            a = -t
            r = int((1 - a) * 255 + a * 49)
            g = int((1 - a) * 255 + a * 130)
            b = int((1 - a) * 255 + a * 189)
        else:
            a = t
            r = int((1 - a) * 255 + a * 202)
            g = int((1 - a) * 255 + a * 0)
            b = int((1 - a) * 255 + a * 32)
        return f"#{r:02x}{g:02x}{b:02x}"
    t = (value - vmin) / max(vmax - vmin, 1e-12)
    t = max(0.0, min(1.0, t))
    r = int((1 - t) * 247 + t * 8)
    g = int((1 - t) * 251 + t * 81)
    b = int((1 - t) * 255 + t * 156)
    return f"#{r:02x}{g:02x}{b:02x}"


def _heatmap_svg(parts: list[str], matrix: np.ndarray, x: int, y: int, title: str,
                 vmin: float, vmax: float, *, diverging: bool = False):
    cell = 24
    label_w = 42
    parts.append(f'<text x="{x}" y="{y - 16}" font-size="16" font-weight="700">{title}</text>')
    for h in range(matrix.shape[1]):
        parts.append(f'<text x="{x + label_w + h * cell + 8}" y="{y - 2}" font-size="10">h{h}</text>')
    for layer in range(matrix.shape[0]):
        parts.append(f'<text x="{x + 4}" y="{y + layer * cell + 16}" font-size="10">L{layer}</text>')
        for h in range(matrix.shape[1]):
            color = _hex_color(float(matrix[layer, h]), vmin, vmax, diverging=diverging)
            parts.append(
                f'<rect x="{x + label_w + h * cell}" y="{y + layer * cell}" width="{cell}" height="{cell}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )


def _line_svg(parts: list[str], series: list[tuple[str, np.ndarray, str]], x: int, y: int,
              width: int, height: int, title: str):
    parts.append(f'<text x="{x}" y="{y - 16}" font-size="16" font-weight="700">{title}</text>')
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#c7c7c7"/>')
    finite_groups = [s[np.isfinite(s)] for _, s, _ in series if np.isfinite(s).any()]
    finite_vals = np.concatenate(finite_groups) if finite_groups else np.array([0.0])
    ymax = float(max(finite_vals.max(initial=0.0), 1e-9)) * 1.08
    for tick in range(5):
        yy = y + height - tick * height / 4
        val = tick * ymax / 4
        parts.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + width}" y2="{yy:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x - 44}" y="{yy + 4:.1f}" font-size="10">{val:.3g}</text>')
    for label, values, color in series:
        pts = []
        for layer, val in enumerate(values):
            if not np.isfinite(val):
                continue
            xx = x + layer * width / max(len(values) - 1, 1)
            yy = y + height - float(val) * height / ymax
            pts.append(f"{xx:.1f},{yy:.1f}")
        if len(pts) >= 2:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for pt in pts:
            xx, yy = pt.split(",")
            parts.append(f'<circle cx="{xx}" cy="{yy}" r="3" fill="{color}"/>')
    legend_x = x + width - 160
    for i, (label, _, color) in enumerate(series):
        yy = y + 18 + i * 18
        parts.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x + 18}" y2="{yy}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{yy + 4}" font-size="12">{label}</text>')
    for layer in range(len(series[0][1])):
        xx = x + layer * width / max(len(series[0][1]) - 1, 1)
        parts.append(f'<text x="{xx - 5:.1f}" y="{y + height + 18}" font-size="10">{layer}</text>')


def save_xsa_interp_outputs(data: dict, save_dir: str, run_id: str):
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"{run_id}_xsa_geometry.npz")
    np.savez(npz_path, **data)

    finite_cos = data["cos_y_v"][np.isfinite(data["cos_y_v"])]
    cos_lim = float(max(abs(finite_cos.min(initial=0.0)), abs(finite_cos.max(initial=0.0)), 1e-6))
    removed_max = float(np.nanmax(data["removed_frac_value"])) if np.isfinite(data["removed_frac_value"]).any() else 1.0
    inc_max = float(np.nanmax(data["model_span_increment"])) if np.isfinite(data["model_span_increment"]).any() else 1.0
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="980" viewBox="0 0 1040 980">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-size="22" font-weight="800">XSA Geometry Diagnostics</text>',
    ]
    _heatmap_svg(parts, data["cos_y_v"], 42, 92, "1. Head-space self-alignment: cos(y, v)",
                 -cos_lim, cos_lim, diverging=True)
    _heatmap_svg(parts, data["removed_frac_value"], 360, 92, "2. Removed energy fraction",
                 0.0, max(removed_max, 1e-9))
    _line_svg(
        parts,
        [
            ("3. residual leakage", data["resid_frac"], "#ca0020"),
            ("4. model-span energy", data["model_span_frac"], "#0571b0"),
        ],
        70,
        430,
        520,
        210,
        "Residual/model-space energy by layer",
    )
    _heatmap_svg(parts, data["model_span_increment"], 680, 430, "Model-span increments by head",
                 0.0, max(inc_max, 1e-9))
    _heatmap_svg(parts, data["attn_gate_mean"], 42, 720, "Attention gate mean",
                 0.0, max(float(np.nanmax(data["attn_gate_mean"])), 1e-9))
    _line_svg(
        parts,
        [
            ("raw head norm", np.nanmean(data["y_norm"], axis=1), "#404040"),
            ("post-head-XSA norm", np.nanmean(data["y_post_head_xsa_norm"], axis=1), "#ca0020"),
            ("gated head norm", np.nanmean(data["y_gated_norm"], axis=1), "#0571b0"),
        ],
        360,
        720,
        520,
        170,
        "Mean head-output norm by layer",
    )
    _line_svg(
        parts,
        [
            ("pre model-XSA", data["o_pre_model_xsa_norm"], "#ca0020"),
            ("post XSA", data["o_post_xsa_norm"], "#0571b0"),
        ],
        70,
        900,
        820,
        55,
        "Post-Wo output norm by layer",
    )
    parts.append("</svg>")

    svg_path = os.path.join(save_dir, f"{run_id}_xsa_geometry.svg")
    with open(svg_path, "w") as f:
        f.write("\n".join(parts))

    return npz_path, svg_path


def save_xsa_alignment_sweep_outputs(results: list[dict], save_dir: str, run_id: str):
    os.makedirs(save_dir, exist_ok=True)
    modes = np.array([str(np.asarray(data["xsa_mode"]).item()) for data in results])
    stack_keys = [
        "head_count",
        "token_count",
        "cos_y_v",
        "removed_frac_value",
        "cos_y_post_v",
        "removed_frac_value_post",
        "y_norm",
        "y_post_head_xsa_norm",
        "y_gated_norm",
        "attn_gate_mean",
        "attn_gate_std",
        "cos_o_x_pre",
        "resid_frac_pre",
        "cos_o_x_post",
        "resid_frac_post",
        "o_pre_model_xsa_norm",
        "o_post_xsa_norm",
        "model_span_frac_pre",
        "model_span_increment_pre",
        "model_span_frac_post",
        "model_span_increment_post",
        "resid_frac",
        "model_span_frac",
        "model_span_increment",
    ]
    combined = {"xsa_modes": modes, "interp_test": np.array("alignment")}
    for key in stack_keys:
        combined[key] = np.stack([data[key] for data in results], axis=0)
    for key in ("model_xsa_lambda", "model_xsa_gram_schmidt", "output_metric_detach",
                "xsa_interp_batches", "xsa_interp_batch_size"):
        if key in results[0]:
            combined[key] = np.asarray(results[0][key])

    npz_path = os.path.join(save_dir, f"{run_id}_alignment_sweep_xsa_geometry.npz")
    np.savez(npz_path, **combined)

    mode_count = len(results)
    row_h = 28
    col_w = 150
    height = 170 + row_h * max(mode_count, 1)
    metrics = [
        ("raw mean cos(y,v)", lambda d: np.nanmean(d["cos_y_v"])),
        ("post mean cos(y,v)", lambda d: np.nanmean(d["cos_y_post_v"])),
        ("raw value frac", lambda d: np.nanmean(d["removed_frac_value"])),
        ("post value frac", lambda d: np.nanmean(d["removed_frac_value_post"])),
        ("gate mean", lambda d: np.nanmean(d["attn_gate_mean"])),
        ("gated norm", lambda d: np.nanmean(d["y_gated_norm"])),
        ("post Wo norm", lambda d: np.nanmean(d["o_post_xsa_norm"])),
        ("post resid frac", lambda d: np.nanmean(d["resid_frac_post"])),
        ("post model-span frac", lambda d: np.nanmean(d["model_span_frac_post"])),
    ]
    width = max(1180, 230 + col_w * len(metrics))
    values = np.array([[fn(data) for _, fn in metrics] for data in results], dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(finite.max(initial=1e-9)) if finite.size else 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-size="22" font-weight="800">XSA Alignment Sweep</text>',
        '<text x="32" y="66" font-size="12" fill="#444">Rows are XSA modes; columns summarize alignment before and after each mode.</text>',
    ]
    x0 = 210
    y0 = 116
    parts.append(f'<text x="32" y="{y0 - 18}" font-size="13" font-weight="700">mode</text>')
    for col, (label, _) in enumerate(metrics):
        parts.append(f'<text x="{x0 + col * col_w}" y="{y0 - 18}" font-size="12" font-weight="700">{label}</text>')
    for row, mode in enumerate(modes):
        y = y0 + row * row_h
        parts.append(f'<text x="32" y="{y + 18}" font-size="12">{mode}</text>')
        for col in range(len(metrics)):
            val = float(values[row, col])
            color = _hex_color(val, 0.0, vmax)
            x = x0 + col * col_w
            parts.append(f'<rect x="{x}" y="{y}" width="136" height="22" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + 8}" y="{y + 15}" font-size="11">{val:.5g}</text>')
    parts.append("</svg>")

    svg_path = os.path.join(save_dir, f"{run_id}_alignment_sweep_xsa_geometry.svg")
    with open(svg_path, "w") as f:
        f.write("\n".join(parts))

    return npz_path, svg_path


def save_xsa_loss_slice_outputs(data: dict, save_dir: str, run_id: str):
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"{run_id}_loss_slices_xsa_geometry.npz")
    np.savez(npz_path, **data)

    modes = [str(x) for x in data["mode_names"]]
    buckets = [str(x) for x in data["bucket_names"]]
    counts = data["bucket_count"]
    delta_mean = data["delta_loss_mean"]
    delta_sum = data["delta_loss_sum"]
    positive_frac = data["positive_delta_frac"]
    compare_mode_idxs = list(range(1, len(modes))) or [0]

    row_h = 24
    col_w = 132
    width = max(980, 360 + col_w * len(compare_mode_idxs) * 2)
    height = 150 + row_h * len(buckets)
    finite = delta_mean[compare_mode_idxs]
    finite = finite[np.isfinite(finite)]
    lim = float(max(abs(finite.min(initial=0.0)), abs(finite.max(initial=0.0)), 1e-9))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-size="22" font-weight="800">XSA Loss Slices</text>',
        f'<text x="32" y="66" font-size="12" fill="#444">Baseline mode: {html.escape(str(data["baseline_xsa_mode"]))}. Positive delta means lower loss than baseline.</text>',
    ]
    y0 = 120
    parts.append(f'<text x="32" y="{y0 - 18}" font-size="12" font-weight="700">bucket</text>')
    parts.append(f'<text x="270" y="{y0 - 18}" font-size="12" font-weight="700">count</text>')
    x0 = 350
    for idx, mode_idx in enumerate(compare_mode_idxs):
        x = x0 + idx * col_w * 2
        parts.append(f'<text x="{x}" y="{y0 - 18}" font-size="12" font-weight="700">{html.escape(modes[mode_idx])} mean</text>')
        parts.append(f'<text x="{x + col_w}" y="{y0 - 18}" font-size="12" font-weight="700">sum / win%</text>')

    for row, bucket in enumerate(buckets):
        y = y0 + row * row_h
        fill = "#fbfbfb" if row % 2 == 0 else "#ffffff"
        parts.append(f'<rect x="24" y="{y - 4}" width="{width - 48}" height="{row_h}" fill="{fill}"/>')
        parts.append(f'<text x="32" y="{y + 12}" font-size="11">{html.escape(bucket)}</text>')
        parts.append(f'<text x="270" y="{y + 12}" font-size="11">{int(counts[row])}</text>')
        for idx, mode_idx in enumerate(compare_mode_idxs):
            mean_val = float(delta_mean[mode_idx, row])
            sum_val = float(delta_sum[mode_idx, row])
            win_val = float(positive_frac[mode_idx, row])
            color = _hex_color(mean_val, -lim, lim, diverging=True)
            x = x0 + idx * col_w * 2
            parts.append(f'<rect x="{x}" y="{y - 3}" width="{col_w - 8}" height="18" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + 6}" y="{y + 11}" font-size="10">{mean_val:.5g}</text>')
            parts.append(f'<text x="{x + col_w}" y="{y + 11}" font-size="10">{sum_val:.4g} / {100 * win_val:.1f}%</text>')

    parts.append("</svg>")

    svg_path = os.path.join(save_dir, f"{run_id}_loss_slices_xsa_geometry.svg")
    with open(svg_path, "w") as f:
        f.write("\n".join(parts))

    return npz_path, svg_path


def save_xsa_attention_diag_outputs(data: dict, save_dir: str, run_id: str):
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"{run_id}_attention_diagnostics.npz")
    np.savez(npz_path, **data)

    def finite_max(name: str, default: float = 1.0):
        vals = data[name][np.isfinite(data[name])]
        return float(vals.max(initial=default)) if vals.size else default

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="1160" viewBox="0 0 1180 1160">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="38" font-size="22" font-weight="800">XSA Attention Diagnostics: {html.escape(str(data["xsa_mode"]))}</text>',
    ]
    _heatmap_svg(parts, data["self_mass"], 42, 92, "Self-attention mass aii",
                 0.0, max(finite_max("self_mass"), 1e-9))
    _heatmap_svg(parts, data["attention_entropy"], 360, 92, "Attention entropy H(a)",
                 0.0, max(finite_max("attention_entropy"), 1e-9))
    _heatmap_svg(parts, data["first_token_mass"], 680, 92, "First-token mass",
                 0.0, max(finite_max("first_token_mass"), 1e-9))
    _heatmap_svg(parts, data["diag_removed_frac"], 42, 430, "Diagonal removed fraction",
                 0.0, max(finite_max("diag_removed_frac"), 1e-9))
    _heatmap_svg(parts, data["context_parallel_removed_frac"], 360, 430, "Context-parallel removed fraction",
                 0.0, max(finite_max("context_parallel_removed_frac"), 1e-9))
    _heatmap_svg(parts, data["gate_mean"], 680, 430, "Gate mean",
                 0.0, max(finite_max("gate_mean"), 1e-9))
    _line_svg(
        parts,
        [
            ("diag value norm", np.nanmean(data["diag_value_norm"], axis=1), "#ca0020"),
            ("context value norm", np.nanmean(data["context_value_norm"], axis=1), "#0571b0"),
            ("gated norm", np.nanmean(data["gated_norm"], axis=1), "#404040"),
        ],
        80,
        760,
        520,
        150,
        "Mean norms by layer",
    )
    corr = data["correlations"]
    for idx, name in enumerate(data["corr_names"]):
        values = corr[idx]
        lim = float(max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1e-9)) if np.isfinite(values).any() else 1.0
        _heatmap_svg(parts, values, 680 if idx % 2 else 80, 760 + (idx // 2) * 100,
                     f"corr: {html.escape(str(name))}", -lim, lim, diverging=True)
    parts.append("</svg>")

    svg_path = os.path.join(save_dir, f"{run_id}_attention_diagnostics.svg")
    with open(svg_path, "w") as f:
        f.write("\n".join(parts))
    return npz_path, svg_path


def save_xsa_attention_diag_sweep_outputs(results: list[dict], save_dir: str, run_id: str):
    os.makedirs(save_dir, exist_ok=True)
    modes = np.array([str(np.asarray(data["xsa_mode"]).item()) for data in results])
    stack_keys = [
        "count",
        "self_mass",
        "attention_entropy",
        "first_token_mass",
        "non_diag_mass",
        "diag_value_norm",
        "context_value_norm",
        "removed_frac",
        "diag_removed_frac",
        "context_parallel_removed_frac",
        "gate_mean",
        "gated_norm",
        "correlations",
    ]
    combined = {"xsa_modes": modes, "interp_test": np.array("attention-diagnostics")}
    for key in stack_keys:
        combined[key] = np.stack([data[key] for data in results], axis=0)
    combined["corr_names"] = results[0]["corr_names"]
    for key in ("model_xsa_lambda", "output_metric_detach", "xsa_interp_batches", "xsa_interp_batch_size"):
        if key in results[0]:
            combined[key] = np.asarray(results[0][key])

    npz_path = os.path.join(save_dir, f"{run_id}_attention_diagnostics_sweep.npz")
    np.savez(npz_path, **combined)

    metrics = [
        ("self mass", lambda d: np.nanmean(d["self_mass"])),
        ("entropy", lambda d: np.nanmean(d["attention_entropy"])),
        ("diag norm", lambda d: np.nanmean(d["diag_value_norm"])),
        ("ctx norm", lambda d: np.nanmean(d["context_value_norm"])),
        ("diag removed", lambda d: np.nanmean(d["diag_removed_frac"])),
        ("ctx-parallel removed", lambda d: np.nanmean(d["context_parallel_removed_frac"])),
        ("gate", lambda d: np.nanmean(d["gate_mean"])),
        ("gated norm", lambda d: np.nanmean(d["gated_norm"])),
    ]
    values = np.array([[fn(data) for _, fn in metrics] for data in results], dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(finite.max(initial=1e-9)) if finite.size else 1.0
    row_h = 28
    col_w = 150
    width = max(1180, 230 + col_w * len(metrics))
    height = 170 + row_h * max(len(results), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-size="22" font-weight="800">XSA Attention Diagnostics Sweep</text>',
        '<text x="32" y="66" font-size="12" fill="#444">Rows are XSA modes; values are averages over non-paired layers/heads/tokens.</text>',
    ]
    x0 = 210
    y0 = 116
    parts.append(f'<text x="32" y="{y0 - 18}" font-size="13" font-weight="700">mode</text>')
    for col, (label, _) in enumerate(metrics):
        parts.append(f'<text x="{x0 + col * col_w}" y="{y0 - 18}" font-size="12" font-weight="700">{label}</text>')
    for row, mode in enumerate(modes):
        y = y0 + row * row_h
        parts.append(f'<text x="32" y="{y + 18}" font-size="12">{html.escape(mode)}</text>')
        for col in range(len(metrics)):
            val = float(values[row, col])
            color = _hex_color(val, 0.0, vmax)
            x = x0 + col * col_w
            parts.append(f'<rect x="{x}" y="{y}" width="136" height="22" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + 8}" y="{y + 15}" font-size="11">{val:.5g}</text>')
    parts.append("</svg>")

    svg_path = os.path.join(save_dir, f"{run_id}_attention_diagnostics_sweep.svg")
    with open(svg_path, "w") as f:
        f.write("\n".join(parts))
    return npz_path, svg_path


def save_xsa_beta_diag_outputs(data: dict, save_dir: str, run_id: str):
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"{run_id}_beta_diagnostics.npz")
    np.savez(npz_path, **data)

    buckets = [str(x) for x in data["bucket_names"]]
    important_buckets = [
        "all",
        "input_repeated_in_context",
        "target_seen_in_context",
        "prev_bigram_repeated",
        "prev_trigram_repeated",
        "target_nearest_distance_1_16",
        "target_nearest_distance_17_128",
        "target_nearest_distance_129_512",
        "target_freq_gt_16",
        "target_punctuation",
    ]
    bucket_idxs = [buckets.index(name) for name in important_buckets if name in buckets]
    if not bucket_idxs:
        bucket_idxs = list(range(min(len(buckets), 10)))

    xsa_layers = [1, 3, 4, 7, 8, 10]
    intervened_layers = [3, 4, 7, 8]
    layer_groups = [
        ("xsa layers", xsa_layers),
        ("intervened", intervened_layers),
        ("layer 8", [8]),
    ]
    metrics = [
        ("aii", "self_mass"),
        ("beta", "beta_mean"),
        ("abs beta", "beta_abs_mean"),
        ("rms beta", "beta_rms"),
        ("P(beta>0)", "beta_positive_frac"),
        ("P(|beta|>aii)", "beta_abs_gt_self_mass_frac"),
        ("|beta|/aii", "beta_abs_over_self_mass"),
    ]

    rows = []
    for group_name, layers in layer_groups:
        layer_idx = [layer for layer in layers if layer < data["count"].shape[0]]
        if not layer_idx:
            continue
        count = data["count"][np.ix_(layer_idx, range(data["count"].shape[1]), bucket_idxs)]
        weights = np.where(np.isfinite(count), count, 0.0)
        for metric_label, key in metrics:
            values = data[key][np.ix_(layer_idx, range(data[key].shape[1]), bucket_idxs)]
            weighted = np.nansum(values * weights, axis=(0, 1)) / np.maximum(np.nansum(weights, axis=(0, 1)), 1.0)
            rows.append((group_name, metric_label, weighted))

    finite_vals = np.concatenate([row[2][np.isfinite(row[2])] for row in rows if np.isfinite(row[2]).any()])
    vmax = float(finite_vals.max(initial=1e-9)) if finite_vals.size else 1.0
    row_h = 24
    col_w = 122
    left_w = 220
    top_h = 132
    width = max(1180, left_w + col_w * len(bucket_idxs) + 70)
    height = top_h + row_h * len(rows) + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-size="22" font-weight="800">XSA Beta Diagnostics</text>',
        '<text x="32" y="64" font-size="12" fill="#444">beta = &lt;y_context, v_i&gt; / ||v_i||^2. Values are count-weighted over heads and selected layers.</text>',
    ]
    for col, bucket_idx in enumerate(bucket_idxs):
        label = html.escape(buckets[bucket_idx].replace("target_", "t_").replace("_distance_", "_dist_"))
        x = left_w + col * col_w
        parts.append(f'<text x="{x}" y="{top_h - 18}" font-size="10" transform="rotate(-28 {x},{top_h - 18})">{label}</text>')
    parts.append(f'<text x="32" y="{top_h - 18}" font-size="12" font-weight="700">group / metric</text>')

    for row, (group_name, metric_label, values) in enumerate(rows):
        y = top_h + row * row_h
        fill = "#fbfbfb" if row % 2 == 0 else "#ffffff"
        parts.append(f'<rect x="24" y="{y - 4}" width="{width - 48}" height="{row_h}" fill="{fill}"/>')
        parts.append(f'<text x="32" y="{y + 12}" font-size="10">{html.escape(group_name)} / {html.escape(metric_label)}</text>')
        for col, val in enumerate(values):
            x = left_w + col * col_w
            if np.isfinite(val):
                color = _hex_color(float(val), 0.0, vmax)
                label = f"{float(val):.3g}"
            else:
                color = "#f2f2f2"
                label = "nan"
            parts.append(f'<rect x="{x}" y="{y - 3}" width="{col_w - 8}" height="18" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + 5}" y="{y + 11}" font-size="9">{label}</text>')
    parts.append("</svg>")

    svg_path = os.path.join(save_dir, f"{run_id}_beta_diagnostics.svg")
    with open(svg_path, "w") as f:
        f.write("\n".join(parts))
    return npz_path, svg_path


def save_xsa_causal_intervention_outputs(data: dict, save_dir: str, run_id: str):
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"{run_id}_causal_interventions.npz")
    np.savez(npz_path, **data)

    conditions = [str(x) for x in data["mode_names"]]
    buckets = [str(x) for x in data["bucket_names"]]
    delta_mean = data["delta_loss_mean"]
    positive_frac = data["positive_delta_frac"]
    counts = data["bucket_count"]
    important_buckets = [
        "all",
        "target_seen_in_context",
        "input_repeated_in_context",
        "prev_bigram_repeated",
        "prev_trigram_repeated",
        "target_nearest_distance_1_16",
        "target_nearest_distance_17_128",
        "target_nearest_distance_129_512",
        "target_nearest_distance_513_2048",
        "target_freq_gt_16",
    ]
    bucket_idxs = [buckets.index(name) for name in important_buckets if name in buckets]
    if not bucket_idxs:
        bucket_idxs = list(range(min(len(buckets), 12)))

    compare_idxs = list(range(1, len(conditions))) or [0]
    all_idx = buckets.index("all") if "all" in buckets else 0
    order = sorted(compare_idxs, key=lambda idx: float(delta_mean[idx, all_idx]))
    max_rows = min(len(order), 48)
    condition_idxs = order[:max_rows]

    mat = delta_mean[np.ix_(condition_idxs, bucket_idxs)]
    finite = mat[np.isfinite(mat)]
    lim = float(max(abs(finite.min(initial=0.0)), abs(finite.max(initial=0.0)), 1e-9)) if finite.size else 1.0
    row_h = 24
    col_w = 126
    left_w = 330
    top_h = 128
    width = max(1180, left_w + col_w * len(bucket_idxs) + 80)
    height = top_h + row_h * max_rows + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-size="22" font-weight="800">XSA Causal Interventions</text>',
        '<text x="32" y="64" font-size="12" fill="#444">Delta loss is baseline_record - intervention; negative means the intervention is worse than the learned record baseline.</text>',
    ]
    for col, bucket_idx in enumerate(bucket_idxs):
        label = html.escape(buckets[bucket_idx].replace("target_", "t_").replace("_distance_", "_dist_"))
        x = left_w + col * col_w
        parts.append(f'<text x="{x}" y="{top_h - 18}" font-size="10" transform="rotate(-28 {x},{top_h - 18})">{label}</text>')
    parts.append(f'<text x="32" y="{top_h - 18}" font-size="12" font-weight="700">condition</text>')
    parts.append(f'<text x="{left_w - 58}" y="{top_h - 18}" font-size="12" font-weight="700">all win%</text>')

    for row, condition_idx in enumerate(condition_idxs):
        y = top_h + row * row_h
        fill = "#fbfbfb" if row % 2 == 0 else "#ffffff"
        parts.append(f'<rect x="24" y="{y - 4}" width="{width - 48}" height="{row_h}" fill="{fill}"/>')
        label = html.escape(conditions[condition_idx])
        if len(label) > 44:
            label = label[:41] + "..."
        parts.append(f'<text x="32" y="{y + 12}" font-size="10">{label}</text>')
        parts.append(f'<text x="{left_w - 52}" y="{y + 12}" font-size="10">{100 * float(positive_frac[condition_idx, all_idx]):.1f}</text>')
        for col, bucket_idx in enumerate(bucket_idxs):
            val = float(delta_mean[condition_idx, bucket_idx])
            color = _hex_color(val, -lim, lim, diverging=True)
            x = left_w + col * col_w
            parts.append(f'<rect x="{x}" y="{y - 3}" width="{col_w - 8}" height="18" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + 5}" y="{y + 11}" font-size="9">{val:+.3g}</text>')

    legend_y = height - 42
    parts.append(f'<text x="32" y="{legend_y}" font-size="11" fill="#444">Bucket counts: ')
    parts.append(" ".join(
        f'{html.escape(buckets[idx])}={int(counts[idx])}' for idx in bucket_idxs[:6]
    ))
    parts.append('</text>')
    parts.append("</svg>")

    svg_path = os.path.join(save_dir, f"{run_id}_causal_interventions.svg")
    with open(svg_path, "w") as f:
        f.write("\n".join(parts))
    return npz_path, svg_path
