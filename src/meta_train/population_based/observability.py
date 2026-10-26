"""
Plotting helpers for population-based training.

All functions accept a `Population` instance and return a matplotlib Figure
without saving or showing it. Callers can decide whether to display or write
the figure to disk.
"""
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from openai import OpenAI
from sklearn.decomposition import PCA

from meta_train.population_based.population import (
    BinaryTournamentHistoryEntry,
    TournamentHistoryEntry,
    Genotype,
    Population,
)


def _extract_scores_by_step(population: Population) -> Tuple[List[int], List[float], List[int]]:
    """
    Extract (step, score, step_created_at) triples from binary tournament history.
    """
    steps: List[int] = []
    scores: List[float] = []
    created_steps: List[int] = []

    for entry in population.history:
        if not isinstance(entry, (BinaryTournamentHistoryEntry, TournamentHistoryEntry)):
            continue
        for genotype, score in zip(entry.evaluated_genotypes, entry.scores):
            steps.append(entry.step)
            scores.append(score)
            created_steps.append(genotype.step_created_at)

    return steps, scores, created_steps


def plot_scores_over_steps(population: Population) -> plt.Figure:
    """
    Scatterplot of evaluation scores vs. step with hue for genotype creation step.
    """
    steps, scores, created_steps = _extract_scores_by_step(population)
    if not steps:
        raise ValueError("No BinaryTournamentHistoryEntry data found in population.history")

    fig, ax = plt.subplots(figsize=(8, 5))

    hue_min, hue_max = min(created_steps), max(created_steps)
    sns.scatterplot(
        x=steps,
        y=scores,
        hue=created_steps,
        hue_norm=(hue_min, hue_max),
        palette="viridis",
        ax=ax,
        edgecolor="none",
        alpha=0.85,
        legend=False,
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(hue_min, hue_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Step created at")

    ax.set_xlabel("Population step (evaluation logged)")
    ax.set_ylabel("Score")
    ax.set_title("Scores across steps (colored by creation step)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _collect_survivals(
    population: Population, treat_alive_as_dead: bool
) -> pd.DataFrame:
    """
    Collect survival lengths for each genotype with its mutation operator.
    """
    rows: List[Dict[str, object]] = []
    current_step = population.current_step
    all_genotypes: List[Genotype] = population.get_alive() + population.get_dead()

    for genotype in all_genotypes:
        if genotype.step_died_at is None and not treat_alive_as_dead:
            continue

        end_step = genotype.step_died_at if genotype.step_died_at is not None else current_step
        survival = end_step - genotype.step_created_at
        operator_name = genotype.mutation_operator_name or "unknown"
        rows.append(
            {
                "mutation_operator": operator_name,
                "survival": survival,
            }
        )

    return pd.DataFrame(rows)


def plot_survival_by_mutation_operator(population: Population, treat_alive_as_dead: bool = False) -> plt.Figure:
    """
    Bar plot of average survival length by mutation operator with stderr bars.

    Args:
        population: Population to analyze.
        treat_alive_as_dead: If True, alive individuals are treated as dying at
            current_step; otherwise they are ignored.
    """
    df = _collect_survivals(population, treat_alive_as_dead)
    if df.empty:
        raise ValueError("No genotypes with survival data to plot")

    fig, ax = plt.subplots(figsize=(8, 5))

    order = (
        df.groupby("mutation_operator")["survival"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    sns.barplot(
        data=df,
        x="mutation_operator",
        y="survival",
        hue="mutation_operator",
        order=order,
        hue_order=order,
        dodge=False,
        errorbar="se",
        palette="muted",
        legend=False,
        ax=ax,
    )

    ax.tick_params(axis="x", labelrotation=20)
    ax.set_ylabel("Average survival length (steps)")
    ax.set_title("Survival by mutation operator")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plt.close(fig)
    return fig


def compute_score_standard_error_stats(population: Population) -> Tuple[float, float]:
    """
    Compute the mean score standard error across genotypes and its standard error.

    For each genotype with at least two score history entries, we compute the
    sample standard deviation (ddof=1) of its scores and convert it to a standard
    error (std / sqrt(n)). We then return the mean of these per-genotype standard
    errors and the standard error of that mean.
    """
    standard_errors: List[float] = []
    for genotype in population.get_alive() + population.get_dead():
        if len(genotype.score_history) < 2:
            continue
        scores = [entry.score for entry in genotype.score_history]
        if len(scores) < 2:
            continue
        std = float(np.std(scores, ddof=1))
        se = std / np.sqrt(len(scores))
        standard_errors.append(se)

    if not standard_errors:
        raise ValueError("No genotypes with >=2 score history entries to compute standard error")

    mean_standard_error = float(np.mean(standard_errors))
    if len(standard_errors) == 1:
        standard_error_of_mean = 0.0
    else:
        standard_error_of_mean = float(np.std(standard_errors, ddof=1) / np.sqrt(len(standard_errors)))

    return mean_standard_error, standard_error_of_mean


def plot_population_pca_embedding(
    population: Population,
    embedding_model: str = "text-embedding-3-small",
) -> go.Figure:
    """
    Create an interactive PCA scatter plot of population CFD embeddings.

    Embeds all genotypes (alive and dead) using OpenAI embeddings, projects
    to 2D via PCA, and returns an interactive Plotly figure with hover info.

    Args:
        population: Population to visualize.
        embedding_model: OpenAI embedding model to use.

    Returns:
        Plotly Figure that can be saved with fig.write_html(path).
    """
    all_genotypes: List[Genotype] = population.get_alive() + population.get_dead()
    if not all_genotypes:
        raise ValueError("No genotypes in population to embed")

    # Embed all CFDs
    client = OpenAI()
    cfd_texts = [g.cfd for g in all_genotypes]
    response = client.embeddings.create(model=embedding_model, input=cfd_texts)
    embeddings = np.array([item.embedding for item in response.data])

    # PCA to 2D
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)

    # Extract metadata for each genotype
    step_created_list: List[int] = []
    hover_texts: List[str] = []
    outline_colors: List[str] = []

    for g in all_genotypes:
        step_created_list.append(g.step_created_at)

        # Build hover text with full CFD (with proper linebreaks) and metadata
        cfd_with_breaks = g.cfd.replace("\n", "<br>")

        hover_parts = [
            f"<b>CFD:</b><br>{cfd_with_breaks}",
            f"<b>Mutation Operator:</b> {g.mutation_operator_name or 'N/A'}",
            f"<b>Step Created:</b> {g.step_created_at}",
        ]
        if g.step_died_at is not None:
            hover_parts.append(f"<b>Step Died:</b> {g.step_died_at}")
            outline_colors.append("rgba(0,0,0,0)")  # Transparent outline for dead
        else:
            hover_parts.append("<b>Status:</b> Alive")
            outline_colors.append("red")  # Red outline for alive

        # Show all scores from history
        if g.score_history:
            scores_str = ", ".join(f"{entry.score:.4f}" for entry in g.score_history)
            hover_parts.append(f"<b>Scores:</b> [{scores_str}]")

        hover_texts.append("<br>".join(hover_parts))

    # Create Plotly scatter
    fig = go.Figure(
        data=go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color=step_created_list,
                colorscale="Viridis",
                colorbar=dict(title="Step Created"),
                opacity=0.85,
                line=dict(
                    color=outline_colors,
                    width=2,
                ),
            ),
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Population PCA Embedding (colored by creation step)",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        template="plotly_white",
    )

    return fig


__all__ = [
    "plot_scores_over_steps",
    "plot_survival_by_mutation_operator",
    "compute_score_standard_error_stats",
    "plot_population_pca_embedding",
]

