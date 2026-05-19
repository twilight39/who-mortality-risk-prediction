"""Reusable plotting functions with optional figure saving."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Helper: save figure to disk or display it inline."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a confusion matrix for a classifier.

    Args:
        model: Fitted classifier.
        X_test: Test features.
        y_test: True test labels.
        title: Plot title.
        save_path: Optional path to write the PNG.

    Returns:
        The matplotlib Figure object.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(title)
    save_or_show(fig, save_path)
    return fig


def plot_elbow_method(
    inertias: list[float],
    k_range: range,
    title: str = "Elbow Method for Optimal k",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the Elbow Method curve for K-Means.

    Args:
        inertias: List of inertia values.
        k_range: Corresponding range of *k* values.
        title: Plot title.
        save_path: Optional path to write the PNG.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(k_range))
    ax.grid(True)
    save_or_show(fig, save_path)
    return fig


def plot_cluster_bar_chart(
    cluster_analysis: pd.DataFrame,
    title: str = "Mortality Count by Age Group for Each Cluster",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot cluster centroids as a grouped bar chart.

    Args:
        cluster_analysis: DataFrame with clusters as rows and age groups as columns.
        title: Plot title.
        save_path: Optional path to write the PNG.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    cluster_analysis.T.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Average Mortality Count")
    ax.set_xlabel("Age Group")
    ax.tick_params(axis="x", rotation=45)
    save_or_show(fig, save_path)
    return fig


def plot_dendrogram(
    linked_matrix,
    labels: list[str],
    title: str = "Hierarchical Clustering Dendrogram",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a hierarchical clustering dendrogram.

    Args:
        linked_matrix: Linkage matrix from ``scipy.cluster.hierarchy.linkage``.
        labels: Leaf labels (disease names).
        title: Plot title.
        save_path: Optional path to write the PNG.

    Returns:
        The matplotlib Figure object.
    """
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(15, 10))
    dendrogram(
        linked_matrix,
        orientation="top",
        labels=labels,
        distance_sort="descending",
        show_leaf_counts=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Disease Category")
    ax.set_ylabel("Distance (Ward)")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    save_or_show(fig, save_path)
    return fig


def plot_temporal_trends(
    df: pd.DataFrame,
    x: str = "Year",
    y: str = "Mortality Count",
    hue: str = "Disease_L1",
    title: str = "Mortality Trends by Major Disease Category",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot temporal mortality trends with Seaborn.

    Args:
        df: DataFrame in long format.
        x: Column name for the x-axis.
        y: Column name for the y-axis.
        hue: Column name for the colour grouping.
        title: Plot title.
        save_path: Optional path to write the PNG.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o", linewidth=2.5, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Total Mortality Count")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    save_or_show(fig, save_path)
    return fig


def plot_feature_importances(
    importances: pd.Series,
    title: str = "Feature Importances",
    top_n: int = 15,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a horizontal bar chart of feature importances.

    Args:
        importances: Sorted Series of importance scores.
        title: Plot title.
        top_n: Number of top features to display.
        save_path: Optional path to write the PNG.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.head(top_n).plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    save_or_show(fig, save_path)
    return fig
