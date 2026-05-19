"""Generate static figures for README embedding.

This script demonstrates the ``malaysia_mortality`` package in action and
produces PNGs saved to ``reports/figures/``.
"""

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from malaysia_mortality.data import load_eda_data, load_model_data
from malaysia_mortality.evaluation import get_feature_importances
from malaysia_mortality.features import (
    create_risk_levels,
    encode_features_for_classification,
    encode_features_for_regression,
    split_data,
)
from malaysia_mortality.models import (
    run_hierarchical_clustering,
    run_kmeans_elbow,
    train_kmeans,
    train_random_forest_classifier,
    train_random_forest_regressor,
)
from malaysia_mortality.viz import (
    plot_cluster_bar_chart,
    plot_confusion_matrix,
    plot_dendrogram,
    plot_elbow_method,
    plot_feature_importances,
    plot_temporal_trends,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ── Classification ──
    model_df = load_model_data()
    model_df["Risk_Level"] = model_df["Mortality Count"].apply(create_risk_levels)
    X_clf, y_clf = encode_features_for_classification(model_df)
    X_train, X_test, y_train, y_test = split_data(X_clf, y_clf)
    rf_clf = train_random_forest_classifier(X_train, y_train)

    plot_confusion_matrix(
        rf_clf, X_test, y_test,
        title="Confusion Matrix – Random Forest Classifier",
        save_path=REPORTS_DIR / "confusion_matrix.png",
    )
    importances_clf = get_feature_importances(rf_clf, X_clf.columns)
    plot_feature_importances(
        importances_clf,
        title="Feature Importances – Classification",
        save_path=REPORTS_DIR / "feature_importance_clf.png",
    )

    # ── Regression ──
    X_reg, y_reg = encode_features_for_regression(model_df)
    X_train_r, X_test_r, y_train_r, y_test_r = split_data(X_reg, y_reg)
    rf_reg = train_random_forest_regressor(X_train_r, y_train_r)
    importances_reg = get_feature_importances(rf_reg, X_reg.columns)
    plot_feature_importances(
        importances_reg,
        title="Feature Importances – Regression",
        save_path=REPORTS_DIR / "feature_importance_reg.png",
    )

    # ── Clustering ──
    cluster_data = model_df[(model_df["Year"] == 2021) & (model_df["Sex_Persons"] == 1)].copy()
    mortality_profiles = cluster_data.pivot_table(
        index="Disease_L2", columns="Age Group", values="Mortality Count"
    ).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(mortality_profiles)

    inertias = run_kmeans_elbow(scaled, k_range=range(1, 11))
    plot_elbow_method(
        inertias, k_range=range(1, 11),
        save_path=REPORTS_DIR / "elbow_plot.png",
    )

    _, labels = train_kmeans(scaled, n_clusters=5)
    mortality_profiles["Cluster"] = labels
    cluster_analysis = mortality_profiles.groupby("Cluster").mean()
    # drop the Cluster column from the analysis view before plotting
    plot_cluster_bar_chart(
        cluster_analysis,
        save_path=REPORTS_DIR / "cluster_bar_chart.png",
    )

    linked = run_hierarchical_clustering(scaled)
    plot_dendrogram(
        linked, labels=mortality_profiles.index.tolist(),
        save_path=REPORTS_DIR / "dendrogram.png",
    )

    # ── Temporal trends ──
    eda_df = load_eda_data()
    df_persons = eda_df[eda_df["Sex"] == "Persons"].copy()
    l1_trends = (
        df_persons.groupby(["Year", "Disease_L1"])["Mortality Count"]
        .sum()
        .reset_index()
    )
    l1_trends["Disease_L1"] = l1_trends["Disease_L1"].str.strip()
    plot_temporal_trends(
        l1_trends, x="Year", y="Mortality Count", hue="Disease_L1",
        title="Mortality Trends by Major Disease Category in Malaysia",
        save_path=REPORTS_DIR / "temporal_trends.png",
    )

    print(f"Figures saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
