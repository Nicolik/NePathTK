import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def compute_curves_v2(
    y,                     # array-like (n,)
    y_score,               # array-like (n, C) with per-class scores/probs
    *,
    task="roc",            # "roc" or "pr"
    classes_encoding=None, # list[str] or None
    disable_title=False,
    meta=None,             # dict with tags ({"model": "...", "split": "...", ...})
    ignore=None,           # NEW: list/tuple/set of class names OR indices to exclude
    auto_binary_positive_only=True  # NEW: if C==2, keep only class 1
):
    """
    Returns a list of dicts (one per INCLUDED class):
      {
        "curve_type": "roc"|"pr",
        "class_id": int,          # 0..C-1
        "class_name": str,        # label for legend
        "x": np.ndarray,          # FPR (ROC) | Recall (PR)
        "y": np.ndarray,          # TPR (ROC) | Precision (PR)
        "thresholds": np.ndarray,
        "auc": float,             # AUC or AP
        "meta": dict,
      }
    """
    meta = dict(meta or {})
    y = np.asarray(y, dtype=np.int64)
    y_ohe = np.zeros((y.size, y.max() + 1))
    y_ohe[np.arange(y.size), y] = 1
    C = y_ohe.shape[1]

    if classes_encoding is None:
        classes_encoding = [str(i) for i in range(C)]

    # --- figure out which classes to include ---
    include_idxs = set(range(C))

    # auto skip negative in binary problems
    if auto_binary_positive_only and C == 2:
        include_idxs = {1}

    # apply ignore list (names or indices)
    if ignore:
        # normalize ignore entries to indices
        ignore_idxs = set()
        for it in ignore:
            if isinstance(it, (int, np.integer)):
                ignore_idxs.add(int(it))
            else:
                # treat as name; remove all matches
                ignore_idxs.update(i for i, name in enumerate(classes_encoding) if name == it)
        include_idxs -= ignore_idxs

    include_idxs = sorted(include_idxs)

    curves = []
    for i in include_idxs:
        yi_true = y_ohe[:, i]
        yi_score = y_score[:, i]

        prevalence = yi_true.mean()
        meta["binary"] = (C == 2)
        meta["prevalence"] = prevalence

        if task == "pr":
            precision, recall, th = precision_recall_curve(yi_true, yi_score)
            ap = average_precision_score(yi_true, yi_score)
            curve = {
                "curve_type": "pr",
                "class_id": i,
                "class_name": classes_encoding[i],
                "x": recall,         # PR x-axis = recall
                "y": precision,      # PR y-axis = precision
                "thresholds": th,
                "auc": float(ap),
                "meta": {**meta},
            }
        else:
            fpr, tpr, th = roc_curve(yi_true, yi_score)
            roc_auc = auc(fpr, tpr)
            curve = {
                "curve_type": "roc",
                "class_id": i,
                "class_name": classes_encoding[i],
                "x": fpr,            # ROC x-axis = FPR
                "y": tpr,            # ROC y-axis = TPR
                "thresholds": th,
                "auc": float(roc_auc),
                "meta": {**meta}
            }

        curves.append(curve)

    return curves


def plot_curves(
    curves,
    *,
    title=None,
    ax=None,
    legend_loc=None,
    draw_baseline=True,
    style_fn=None,
    save_path=None,
    disable_title=False,
    dpi=300,
    upscale_factor=1.5,
    upscale_line=1.2,
    percent_multiplier=1,
    ndigits=3,
):
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    if not curves:
        raise ValueError("No curves to plot.")

    curve_type = curves[0]["curve_type"]
    if any(c["curve_type"] != curve_type for c in curves):
        raise ValueError("All curves must share the same curve_type for a single plot.")

    if draw_baseline:
        if curve_type == "roc":
            ax.plot([0, 1], [0, 1], "k--", linewidth=2 * upscale_line)
        elif curve_type == "pr":
            for c in curves:
                prevalence = c["meta"].get("prevalence")
                if prevalence is not None:
                    style = style_fn(c) if style_fn else {}
                    ax.plot([0, 1], [prevalence, prevalence],
                            linestyle="--",
                            color=style.get("color", "k"),
                            linewidth=2 * upscale_line)

    auc_text = "AUC"
    if curve_type == "roc":
        auc_text = "AUROC"
    elif curve_type == "pr":
        auc_text = "AUPRC"

    for c in curves:
        x, y = c["x"], c["y"]
        meta = c.get("meta", {})
        split = meta.get("split", "Unknown")
        binary = meta.get("binary", False)

        # --- label logic ---
        if binary:
            label = f"{split} ({auc_text}: {round(c['auc'] * percent_multiplier, ndigits):.{ndigits}f})"
        else:
            # multi-class: keep class name + model/split if available
            parts = []
            if "model" in meta: parts.append(meta["model"])
            if "split" in meta: parts.append(meta["split"])
            suffix = " / ".join(parts) if parts else None
            label = (f"{c['class_name']}" + (f" | {suffix}" if suffix else "") +
                     f" ({auc_text}: {round(c['auc']*percent_multiplier, ndigits)}:.{ndigits}f)")

        style = {"linewidth": 3*upscale_line}
        if style_fn is not None:
            style.update(style_fn(c) or {})
        if "label" not in style:
            style["label"] = label

        ax.plot(x, y, **style)

    # axis labels
    if curve_type == "pr":
        ax.set_xlabel("Recall", fontsize=16*upscale_factor)
        ax.set_ylabel("Precision", fontsize=16*upscale_factor)
        if legend_loc is None:
            legend_loc = "lower left"
        default_title = "Precision–Recall"
    else:
        ax.set_xlabel("False Positive Rate", fontsize=16*upscale_factor)
        ax.set_ylabel("True Positive Rate", fontsize=16*upscale_factor)
        if legend_loc is None:
            legend_loc = "lower right"
        default_title = "ROC"

    # --- Title with description ---
    if title is None:
        # collect unique model/split pairs
        desc_parts = []
        for c in curves:
            meta = c.get("meta", {})
            model = meta.get("model")
            split = meta.get("split")
            if model and split:
                desc_parts.append(f"{model} [{split}]")
            elif split:
                desc_parts.append(f"{split}")
            elif model:
                desc_parts.append(model)
        desc = ", ".join(sorted(set(desc_parts)))
        title = f"{default_title} – {desc}" if desc else default_title

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    if not disable_title:
        ax.set_title(title, fontsize=16*upscale_factor)
    ax.legend(loc=legend_loc, fontsize=14*upscale_factor)
    ax.tick_params(axis='both', labelsize=14*upscale_factor)

    if own_ax:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi)
            plt.close(fig)
        return ax
    else:
        if save_path:
            fig.savefig(save_path, dpi=dpi)
        return ax


def default_style_fn(c):
    # pick color by split
    split = c["meta"].get("split", "Unknown")
    color_map = {
        "Val": "tab:orange",
        "Test": "tab:blue",
        "Train": "tab:green"
    }
    style = {
        "color": color_map.get(split, "tab:gray"),  # fallback gray if not mapped
        "linestyle": "-",                           # always solid
    }
    return style
