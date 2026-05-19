import json
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb
import os
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy

# --- ACADEMIC PLOT STYLE CONFIGURATION ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.weight": "bold",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "axes.linewidth": 1.5,
        "grid.linewidth": 1.0,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)

FIGURES_DIR = "/path/to/figures_pdf"

# --- METRIC MAPPING ---
METRIC_MAP = {
    "Fluency1": "Narrative Pacing",
    "Fluency2": "Scene vs Exposition",
    "Fluency3": "Language Proficiency & Literary Devices",
    "Fluency4": "Narrative Ending",
    "Fluency5": "Understandability & Coherence",
    "Flexibility1": "Perspective & Voice Flexibility",
    "Flexibility2": "Emotional Flexibility",
    "Flexibility3": "Structural Flexibility",
    "Originality1": "Originality in Theme and Content",
    "Originality2": "Originality in Thought",
    "Originality3": "Originality in Form",
    "Elaboration1": "World Building and setting",
    "Elaboration2": "Character Development",
    "Elaboration3": "Rhetorical Complexity",
}

SCORE_MIN = 0
SCORE_MAX = 10

# --- SHORT AXIS LABELS for heatmaps ---
# Derived from METRIC_MAP keys: Fluency->F, Flexibility->Fl, Originality->O, Elaboration->E
# e.g. "Narrative Pacing" (Fluency1) -> "F1", "Emotional Flexibility" (Flexibility2) -> "Fl2"
_METRIC_KEY_ABBREV = {
    "Fluency": "F",
    "Flexibility": "Fl",
    "Originality": "O",
    "Elaboration": "E",
}
METRIC_AXIS_LABELS: dict[str, str] = {}
for _key, _formal in METRIC_MAP.items():
    for _prefix, _abbr in _METRIC_KEY_ABBREV.items():
        if _key.startswith(_prefix):
            METRIC_AXIS_LABELS[_formal] = _abbr + _key[len(_prefix) :]
            break
    else:
        METRIC_AXIS_LABELS[_formal] = _key  # fallback

# --- MODEL DISPLAY NAMES ---
# Maps normalized model names (last path component) to clean x-axis labels.
MODEL_DISPLAY_NAMES = {
    "Llama-3_3-Nemotron-Super-49B-v1_5": "Nemotron-49B",
    "gpt-oss-120b": "GPT-OSS-120B",
    "Qwen3-Next-80B-A3B-Instruct": "Qwen3-80B",
}


def normalize_model_name(model_name: object) -> str:
    if not isinstance(model_name, str):
        return "UNKNOWN_MODEL"
    # Take the last non-empty component so trailing slashes don't produce "".
    parts = [p for p in model_name.strip().split("/") if p]
    return parts[-1] if parts else "UNKNOWN_MODEL"


def normalize_score(score: object) -> float | None:
    if isinstance(score, (int, float)):
        value = float(score)
    elif isinstance(score, str):
        s = score.strip()
        if not s:
            return None
        try:
            value = float(s)
        except ValueError:
            return None
    else:
        return None

    if not np.isfinite(value):
        return None
    if SCORE_MIN <= value <= SCORE_MAX:
        return value
    return None


# ------------------------------------------------------------------ #
# Analysis helpers                                                     #
# ------------------------------------------------------------------ #

SCORE_BINS = list(range(int(SCORE_MIN), int(SCORE_MAX) + 2))  # [0,1,...,11]


def compute_discrimination_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model discrimination quality:
      - mean_variance : average score variance across all (metric, record) pairs
      - score_entropy : entropy of the score-bin histogram (higher = more spread)
      - bin_coverage  : fraction of score bins [0-10] actually used
    Higher values indicate the model spreads its scores more (discriminates better).
    """
    rows = []
    for model, gdf in df.groupby("Model"):
        scores = gdf["Score"].dropna().values
        # Variance across all scores given by this model
        mean_var = float(np.var(scores))
        # Per-metric variance, then averaged
        per_metric_var = gdf.groupby("Metric")["Score"].var().mean()
        # Entropy of score histogram
        counts, _ = np.histogram(scores, bins=SCORE_BINS)
        probs = counts / counts.sum() if counts.sum() > 0 else counts
        ent = float(scipy_entropy(probs + 1e-12))  # add epsilon to avoid log(0)
        max_entropy = float(np.log(len(SCORE_BINS) - 1))
        norm_entropy = ent / max_entropy if max_entropy > 0 else 0.0
        # Bin coverage
        bin_coverage = float((counts > 0).sum() / (len(SCORE_BINS) - 1))
        rows.append(
            {
                "Model": model,
                "Mean_Variance": round(mean_var, 4),
                "PerMetric_Variance": round(float(per_metric_var), 4),
                "Score_Entropy": round(ent, 4),
                "Norm_Entropy": round(norm_entropy, 4),
                "Bin_Coverage": round(bin_coverage, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("Norm_Entropy", ascending=False)


def compute_metric_isolation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model metric-isolation quality:
      - mean_inter_metric_corr : mean absolute Pearson correlation across all
        pairs of metrics (lower = better isolation / more independent judgements)
      - isolation_score        : 1 - mean_inter_metric_corr  (higher = better)
    We pivot to (story x metric) for each model, then compute the correlation matrix.
    """
    rows = []
    for model, gdf in df.groupby("Model"):
        pivot = gdf.pivot_table(
            index="record_idx" if "record_idx" in gdf.columns else gdf.index,
            columns="Metric",
            values="Score",
            aggfunc="mean",
        )
        if pivot.shape[1] < 2:
            continue
        corr_matrix = pivot.corr().abs()
        # Upper triangle only (excluding diagonal)
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        upper = corr_matrix.values[mask]
        mean_corr = float(np.nanmean(upper))
        rows.append(
            {
                "Model": model,
                "Mean_InterMetric_Corr": round(mean_corr, 4),
                "Isolation_Score": round(1.0 - mean_corr, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("Isolation_Score", ascending=False)


def _bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    color_map: dict,
    figures_dir: str,
) -> str:
    """Save a bar chart and return the PNG path."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    display_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in df[x_col]]
    colors = [color_map.get(m, "#a6cee3") for m in df[x_col]]
    bars = ax.bar(display_labels, df[y_col].values, color=colors, edgecolor="black")

    # Label each bar; flip padding direction for negative bars
    for bar, val in zip(bars, df[y_col].values):
        pad = 3 if val >= 0 else -12
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad * 0.003,
            f"{val:.3f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    # No title — caption will be added in the LaTeX source.
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    y_min = df[y_col].min()
    y_max = df[y_col].max()
    padding = max(abs(y_max), abs(y_min)) * 0.25 + 0.01
    ax.set_ylim(min(0, y_min) - padding, max(0, y_max) + padding)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right")
    safe = title.replace(" ", "_").replace("/", "_").replace(":", "")
    png_path = os.path.join(figures_dir, f"{safe}.png")
    pdf_path = png_path.replace(".png", ".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def analyze_and_log(input_path, wandb_project="Story-Evaluation-Analysis"):
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Initialize W&B only after confirming the file exists.
    wandb.init(project=wandb_project, name="model_preference_run")

    data = []
    print(f"Loading and cleaning data from {input_path}...")

    record_idx = 0
    with open(input_path, "rb", buffering=4 * 1024 * 1024) as f:
        for line in tqdm(f, desc="records", unit="row"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for code, formal_name in METRIC_MAP.items():
                metric_data = record.get(code)
                if not isinstance(metric_data, dict):
                    continue
                for model_name, result in metric_data.items():
                    if not isinstance(result, dict):
                        continue
                    # Skip the synthesized "overall" entry
                    if model_name == "overall" or model_name.startswith("_"):
                        continue
                    score = normalize_score(result.get("score"))
                    # Clean data: keep only scores within configured range.
                    if score is not None:
                        data.append(
                            {
                                "record_idx": record_idx,
                                "Metric": formal_name,
                                "Model": normalize_model_name(model_name),
                                "Score": score,
                            }
                        )
            record_idx += 1

    df = pd.DataFrame(data, copy=False)
    if df.empty:
        raise ValueError(
            "No valid scores found after cleaning. Check score format/range in input JSONL."
        )

    print("\n--- Raw model counts (before complete-pairs filter) ---")
    print(df["Model"].value_counts().to_string())
    print("-------------------------------------------------------\n")

    # Keep only records where EVERY model has a valid score for EVERY metric.
    n_models = df["Model"].nunique()
    n_metrics = df["Metric"].nunique()
    before_records = df["record_idx"].nunique()

    combo_counts = (
        df.groupby(["record_idx", "Metric", "Model"])
        .size()
        .groupby(level="record_idx")
        .count()
    )
    complete_records = combo_counts[combo_counts == n_metrics * n_models].index

    df = df[df["record_idx"].isin(complete_records)]  # keep record_idx for analysis
    after_records = len(complete_records)
    print(
        f"Filtered {before_records - after_records} incomplete records. "
        f"{after_records} / {before_records} records retained "
        f"(all {n_models} models scored all {n_metrics} metrics)."
    )
    print("\n--- Model counts after complete-pairs filter ---")
    print(df["Model"].value_counts().to_string())
    print("------------------------------------------------\n")

    # 2. Plotting loop
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating PDF figures for paper...")
    pdf_artifact = wandb.Artifact("figures_pdf", type="figures")

    # Consistent model color mapping across all plots
    all_models = sorted(df["Model"].unique())
    cmap = plt.get_cmap("tab10")
    model_colors = {model: cmap(i % cmap.N) for i, model in enumerate(all_models)}

    # ------------------------------------------------------------------ #
    # A. Discrimination scores                                             #
    # ------------------------------------------------------------------ #
    print("\n--- Discrimination Scores ---")
    disc_df = compute_discrimination_scores(df)
    print(disc_df.to_string(index=False))
    disc_df.to_csv(os.path.join(FIGURES_DIR, "discrimination_scores.csv"), index=False)
    wandb.log({"discrimination_scores": wandb.Table(dataframe=disc_df)})

    for y_col, title, ylabel in [
        (
            "Norm_Entropy",
            "Discrimination: Normalised Score Entropy",
            "Normalised Entropy",
        ),
        ("Bin_Coverage", "Discrimination: Score Bin Coverage", "Fraction of Bins Used"),
        ("PerMetric_Variance", "Discrimination: Per-Metric Score Variance", "Variance"),
    ]:
        png_path, pdf_path = _bar_plot(
            disc_df, "Model", y_col, title, ylabel, model_colors, FIGURES_DIR
        )
        wandb.log({f"Discrimination/{title}": wandb.Image(png_path)}, commit=False)
        wandb.save(pdf_path, base_path=FIGURES_DIR, policy="now")
        pdf_artifact.add_file(pdf_path)

    # ------------------------------------------------------------------ #
    # B. Metric-isolation scores                                           #
    # ------------------------------------------------------------------ #
    print("\n--- Metric-Isolation Scores ---")
    iso_df = compute_metric_isolation_scores(df)
    print(iso_df.to_string(index=False))
    iso_df.to_csv(os.path.join(FIGURES_DIR, "metric_isolation_scores.csv"), index=False)
    wandb.log({"metric_isolation_scores": wandb.Table(dataframe=iso_df)})

    png_path, pdf_path = _bar_plot(
        iso_df,
        "Model",
        "Isolation_Score",
        "Metric-Isolation Score (1 - Mean Inter-Metric Corr)",
        "Isolation Score (higher = better)",
        model_colors,
        FIGURES_DIR,
    )
    wandb.log({"Isolation/Metric_Isolation_Score": wandb.Image(png_path)}, commit=False)
    wandb.save(pdf_path, base_path=FIGURES_DIR, policy="now")
    pdf_artifact.add_file(pdf_path)

    # Also plot heatmaps of the inter-metric correlation matrices per model
    _heatmap_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.weight": "normal",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
    for model, gdf in df.groupby("Model"):
        pivot = gdf.pivot_table(
            index="record_idx", columns="Metric", values="Score", aggfunc="mean"
        )
        corr_matrix = pivot.corr()
        n = len(corr_matrix)

        with plt.rc_context(_heatmap_rc):
            # Fixed size suitable for a full text-width figure in a paper.
            fig, ax = plt.subplots(figsize=(6.3, 5.5))

            # Reorder to match METRIC_MAP definition order before plotting.
            ordered_labels = [
                name for name in METRIC_MAP.values() if name in corr_matrix.columns
            ]
            corr_matrix = corr_matrix.loc[ordered_labels, ordered_labels]

            im = ax.imshow(
                corr_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal"
            )
            cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            cbar.set_label("Pearson r")
            cbar.ax.tick_params()

            # Annotate each cell; use a small font so "-0.xx" never overflows.
            for i in range(n):
                for j in range(n):
                    val = corr_matrix.values[i, j]
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="black" if abs(val) < 0.65 else "white",
                    )

            # Draw a thick border around each same-dimension block on the diagonal.
            # Compute contiguous group boundaries from METRIC_MAP key prefixes.
            _dim_colors = {
                "Fluency": "#1f77b4",
                "Flexibility": "#ff7f0e",
                "Originality": "#2ca02c",
                "Elaboration": "#d62728",
            }
            _dim_order = ["Fluency", "Flexibility", "Originality", "Elaboration"]
            _dim_sizes = {d: sum(1 for k in METRIC_MAP if k.startswith(d)) for d in _dim_order}
            _start = 0
            for _dim in _dim_order:
                _sz = _dim_sizes.get(_dim, 0)
                if _sz == 0:
                    continue
                _color = _dim_colors[_dim]
                rect = mpatches.FancyBboxPatch(
                    (_start - 0.5, _start - 0.5), _sz, _sz,
                    boxstyle="square,pad=0",
                    linewidth=2.0, edgecolor=_color, facecolor="none",
                    zorder=3,
                )
                ax.add_patch(rect)
                _start += _sz

            # Draw light separator lines between dimensions.
            _start = 0
            for _dim in _dim_order[:-1]:
                _start += _dim_sizes.get(_dim, 0)
                ax.axhline(_start - 0.5, color="white", linewidth=1.0, zorder=2)
                ax.axvline(_start - 0.5, color="white", linewidth=1.0, zorder=2)

            # Legend for dimension colours — placed below the figure.
            # bbox_inches="tight" on save ensures it is not clipped.
            legend_handles = [
                mpatches.Patch(edgecolor=_dim_colors[d], facecolor="none", linewidth=2.0, label=d)
                for d in _dim_order if _dim_sizes.get(d, 0) > 0
            ]
            fig.legend(handles=legend_handles, loc="lower center",
                       bbox_to_anchor=(0.42, -0.04), ncol=4, fontsize=7,
                       framealpha=0.8, borderpad=0.5, handlelength=1.5)

            _WRAP = 14
            x_labels = [textwrap.fill(l, width=_WRAP) for l in ordered_labels]
            y_labels = [textwrap.fill(l, width=_WRAP) for l in ordered_labels]

            ax.set_xticks(range(n))
            ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=6)
            ax.set_yticks(range(n))
            ax.set_yticklabels(y_labels, fontsize=6)
            ax.tick_params(axis="both", which="both", length=0)

            display_model = MODEL_DISPLAY_NAMES.get(model, model)
            ax.set_title(f"Inter-Metric Correlation: {display_model}", pad=14)
            fig.tight_layout(pad=1.5)
            safe = model.replace("/", "_")
            png_path = os.path.join(FIGURES_DIR, f"corr_heatmap_{safe}.png")
            pdf_path = os.path.join(FIGURES_DIR, f"corr_heatmap_{safe}.pdf")
            fig.savefig(png_path, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
        wandb.log(
            {f"Isolation/Corr_Heatmap_{display_model}": wandb.Image(png_path)},
            commit=False,
        )
        wandb.save(pdf_path, base_path=FIGURES_DIR, policy="now")
        pdf_artifact.add_file(pdf_path)

    # ------------------------------------------------------------------ #
    # 1. Generate Statistical Summary Table (drop record_idx here)        #
    # ------------------------------------------------------------------ #
    df = df.drop(columns="record_idx")

    # 1. Generate Statistical Summary Table
    stats = (
        df.groupby(["Metric", "Model"])
        .agg(
            Sample_Size=pd.NamedAgg(column="Score", aggfunc="count"),
            Mean=pd.NamedAgg(column="Score", aggfunc="mean"),
            Std=pd.NamedAgg(column="Score", aggfunc="std"),
            Min=pd.NamedAgg(column="Score", aggfunc="min"),
            Q1=pd.NamedAgg(column="Score", aggfunc=lambda x: x.quantile(0.25)),
            Median=pd.NamedAgg(column="Score", aggfunc="median"),
            Q3=pd.NamedAgg(column="Score", aggfunc=lambda x: x.quantile(0.75)),
            Max=pd.NamedAgg(column="Score", aggfunc="max"),
        )
        .reset_index()
    )

    # Log Table to W&B
    wandb.log({"score_statistics_table": wandb.Table(dataframe=stats)})
    stats.to_csv(os.path.join(FIGURES_DIR, "model_score_analysis.csv"), index=False)

    # RC overrides for small paper figures (≈33 % of line width ≈ 2.4 in).
    # Font sizes are enlarged relative to the tiny canvas so they remain
    # legible after inclusion in a LaTeX document.
    _dist_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.weight": "bold",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
    # Fixed figure size: width ≈ 2.4 in (matches ~33 % of a 17 cm text block),
    # height keeps a square-ish aspect so the box plot uses vertical space well.
    _DIST_FIGSIZE = (2.4, 2.2)

    for metric in df["Metric"].unique():
        subset = df[df["Metric"] == metric]

        # Box plot distribution for each model
        models = sorted(subset["Model"].unique())
        plot_models = []
        plot_data = []
        for model in models:
            values = subset[subset["Model"] == model]["Score"].dropna().values
            if len(values) == 0:
                continue
            plot_models.append(model)
            plot_data.append(values)

        if not plot_data:
            print(f"Skipping {metric}: no valid scores after filtering.")
            continue

        display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in plot_models]
        counts_text = ", ".join(
            [
                f"{MODEL_DISPLAY_NAMES.get(model, model)}: n={len(vals)} "
                f"[{vals.min():.1f}-{vals.max():.1f}]"
                for model, vals in zip(plot_models, plot_data)
            ]
        )
        print(f"{metric} -> {counts_text}")

        with plt.rc_context(_dist_rc):
            fig, ax = plt.subplots(figsize=_DIST_FIGSIZE)

            box = ax.boxplot(
                plot_data,
                tick_labels=display_names,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.5},
                boxprops={"edgecolor": "black", "linewidth": 0.8},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8},
            )

            # Apply consistent colors per model
            for patch, model in zip(box["boxes"], plot_models):
                patch.set_facecolor(model_colors.get(model, "#a6cee3"))

            # No title — caption will be added in the LaTeX source.
            ax.set_xlabel("Model", fontweight="bold")
            ax.set_ylabel(f"Score ({SCORE_MIN}–{SCORE_MAX})", fontweight="bold")
            ax.set_ylim(SCORE_MIN - 0.5, SCORE_MAX + 0.5)
            ax.tick_params(axis="x", rotation=20)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
            ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.6)

            # Save as PDF for paper and PNG for W&B preview
            safe_name = metric.replace(" ", "_").replace("&", "and")
            pdf_path = os.path.join(FIGURES_DIR, f"{safe_name}_distribution.pdf")
            png_path = os.path.join(FIGURES_DIR, f"{safe_name}_distribution.png")
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

        # Log high-res PNG to W&B media panel for quick viewing.
        wandb.log({f"Distributions/{metric}": wandb.Image(png_path)}, commit=False)
        # Save PDF to run Files tab (directly downloadable) and add to artifact.
        wandb.save(pdf_path, base_path=FIGURES_DIR, policy="now")
        pdf_artifact.add_file(pdf_path)

    # Upload all PDFs as a single versioned artifact.
    wandb.log_artifact(pdf_artifact)

    # Commit all buffered figure logs in a single step.
    wandb.log({})

    print(f"Analysis Complete.")
    print(
        f"1. Statistics saved to: {os.path.join(FIGURES_DIR, 'model_score_analysis.csv')}"
    )
    print(
        f"2. Discrimination scores: {os.path.join(FIGURES_DIR, 'discrimination_scores.csv')}"
    )
    print(
        f"3. Metric-isolation scores: {os.path.join(FIGURES_DIR, 'metric_isolation_scores.csv')}"
    )
    print(f"4. PDF figures saved in: {FIGURES_DIR}")
    wandb.finish()


if __name__ == "__main__":
    analyze_and_log("/path/to/story_evaluation_dataset_evaluated.jsonl")
