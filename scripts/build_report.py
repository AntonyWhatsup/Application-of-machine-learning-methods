"""Build a compact, readable PDF report from generated metrics and figures."""

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
OUTPUT = REPORTS / "project_report.pdf"

CAPTIONS = {
    "eda_01_missingness_by_employment": "Braki zależą od statusu zatrudnienia; imputacja musi być uczona na train.",
    "eda_02_all_numeric_distributions": "Salary pokazano w log-scale, staż w zakresie 0–60, a Counter pominięto.",
    "eda_03_salary_hist_kde": "Surowe SalaryUSD jest silnie prawostronnie skośne.",
    "eda_04_salary_log_hist": "log1p stabilizuje rozkład celu regresji.",
    "eda_05_anomalies_boxplots": "Anomalie są widoczne osobno i nie zniekształcają wykresów głównych.",
    "eda_06_salary_top_countries": "Kraj wyraźnie różnicuje medianę wynagrodzenia.",
    "eda_07_salary_top_jobs": "Stanowisko jest ważne, lecz grupy mocno się nakładają.",
    "eda_08_salary_by_year": "Zmiana w czasie uzasadnia zachowanie Survey Year i temporal split.",
    "eda_09_experience_salary_nonlinear": "Zależność stażu i płacy jest dodatnia, nieliniowa i rozproszona.",
    "eda_10_salary_experience_level": "Poziomy doświadczenia różnią mediany, ale nie separują grup.",
    "eda_11_career_by_employment": "Plany zawodowe zależą od statusu zatrudnienia.",
    "eda_12_career_by_experience": "Nierównowaga klas uzasadnia Macro F1 zamiast accuracy.",
    "eda_13_other_databases_frequency": "OtherDatabases wymaga kodowania multi-label.",
    "eda_14_category_frequencies": "Rzadkie kategorie są grupowane wyłącznie na train.",
    "eda_15_spearman_correlations": "Słabe korelacje parami nie wykluczają nieliniowych interakcji.",
    "model_00_combined_overview": "Regresję i klasyfikację oceniamy w ich własnych metrykach.",
    "model_01_regression_comparison": "Model wybrano według CV MAE; test jest oceną końcową.",
    "model_02_regression_importance": "Ważności są predykcyjne, nie przyczynowe.",
    "model_03_classification_comparison": "Macro F1 około 0,25 oznacza niską jakość praktyczną.",
    "model_04_classification_confusion_matrix": "Rzadkie klasy mają niski recall i często są mylone.",
    "model_05_classification_importance": "Model nadaje się do eksploracji, nie decyzji kadrowych.",
    "cluster_01_svd_explained_variance": "Dwa komponenty służą wyłącznie do wizualizacji.",
    "cluster_02_k_selection": "Niskie silhouette wskazuje słabą strukturę klastrów.",
    "cluster_03_kmeans_2d": "K-Means uczono na 20 komponentach; wykres jest tylko rzutem 2D.",
    "cluster_04_profiles": "Profile są opisowe i nie stanowią stabilnych segmentów operacyjnych.",
}

FIGURE_GROUPS = [
    ("EDA: jakość, rozkłady i anomalie", ["eda_01_missingness_by_employment", "eda_02_all_numeric_distributions", "eda_05_anomalies_boxplots"]),
    ("EDA: rozkład wynagrodzenia i czas", ["eda_03_salary_hist_kde", "eda_04_salary_log_hist", "eda_08_salary_by_year"]),
    ("EDA: determinanty wynagrodzenia", ["eda_06_salary_top_countries", "eda_07_salary_top_jobs", "eda_09_experience_salary_nonlinear"]),
    ("EDA: doświadczenie i plany zawodowe", ["eda_10_salary_experience_level", "eda_11_career_by_employment", "eda_12_career_by_experience"]),
    ("EDA: kategorie i zależności", ["eda_13_other_databases_frequency", "eda_14_category_frequencies", "eda_15_spearman_correlations"]),
    ("Modele: przegląd i regresja", ["model_00_combined_overview", "model_01_regression_comparison", "model_02_regression_importance"]),
    ("Modele: klasyfikacja", ["model_03_classification_comparison", "model_04_classification_confusion_matrix", "model_05_classification_importance"]),
    ("Klasteryzacja: redukcja wymiaru i wybór k", ["cluster_01_svd_explained_variance", "cluster_02_k_selection"]),
    ("Klasteryzacja: mapa i profile", ["cluster_03_kmeans_2d", "cluster_04_profiles"]),
]


def text_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=20, weight="bold", va="top")
    y = 0.87
    for paragraph in paragraphs:
        lines = wrap(paragraph, width=92)
        fig.text(0.08, y, "\n".join(lines), fontsize=11, va="top", linespacing=1.45)
        y -= 0.032 * len(lines) + 0.045
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def draw_table(ax, data, title, col_widths=None, fontsize=8.5):
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", pad=10)
    table = ax.table(
        cellText=data.values,
        colLabels=data.columns,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DCE6F1")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F7FA")


def summary_tables_page(pdf):
    regression = pd.read_csv(REPORTS / "regression_model_comparison.csv")
    regression_table = pd.DataFrame({
        "Model": regression["model"],
        "CV MAE ± SD": [f"{m:,.0f} ± {s:,.0f}" for m, s in zip(regression["CV_MAE_USD_mean"], regression["CV_MAE_USD_std"])],
        "Test MAE": regression["test_MAE_USD"].map(lambda x: f"{x:,.0f}"),
        "Test RMSE": regression["test_RMSE_USD"].map(lambda x: f"{x:,.0f}"),
        "Test R² (log)": regression["test_R2_log"].map(lambda x: f"{x:.3f}"),
    })
    classification = pd.read_csv(REPORTS / "classification_model_comparison.csv")
    classification_table = pd.DataFrame({
        "Model": classification["model"],
        "CV Macro F1 ± SD": [f"{m:.3f} ± {s:.3f}" for m, s in zip(classification["CV_macro_F1_mean"], classification["CV_macro_F1_std"])],
        "Test Macro F1": classification["test_macro_F1"].map(lambda x: f"{x:.3f}"),
        "Balanced accuracy": classification["test_balanced_accuracy"].map(lambda x: f"{x:.3f}"),
    })
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.suptitle("Podsumowanie jakości modeli", fontsize=18, weight="bold")
    draw_table(axes[0], regression_table, "Regresja [USD]", [0.28, 0.25, 0.16, 0.16, 0.15])
    draw_table(axes[1], classification_table, "Klasyfikacja", [0.31, 0.27, 0.21, 0.21])
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def detail_tables_page(pdf):
    per_class = pd.read_csv(REPORTS / "classification_per_class_metrics.csv")
    unnamed = [c for c in per_class.columns if c.startswith("Unnamed:")]
    if unnamed:
        per_class = per_class.rename(columns={unnamed[0]: "class"})
    per_class = per_class.loc[per_class["class"] != "accuracy"].copy()
    per_class["class"] = per_class["class"].map(lambda value: "\n".join(wrap(str(value), 34)))
    per_class = per_class.rename(columns={
        "class": "Class", "precision": "Precision", "recall": "Recall",
        "f1-score": "F1", "support": "Support",
    })[["Class", "Precision", "Recall", "F1", "Support"]]
    for column in ["Precision", "Recall", "F1"]:
        per_class[column] = per_class[column].map(lambda x: f"{x:.3f}")
    per_class["Support"] = per_class["Support"].map(lambda x: f"{x:,.0f}")

    clusters = pd.read_csv(REPORTS / "cluster_profiles.csv")
    cluster_table = pd.DataFrame({
        "Cluster": clusters["Cluster_KMeans"].astype(str),
        "N": clusters["size"].map(lambda x: f"{x:,}"),
        "Share": clusters["share_pct"].map(lambda x: f"{x:.1f}%"),
        "Median salary": clusters["median_salary"].map(lambda x: f"${x:,.0f}"),
        "Job years": clusters["median_job_years"].map(lambda x: f"{x:.1f}"),
        "DB years": clusters["median_database_years"].map(lambda x: f"{x:.1f}"),
        "Country": clusters["dominant_country"],
    })
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), gridspec_kw={"height_ratios": [1.65, 1]})
    fig.suptitle("Szczegółowe wyniki", fontsize=18, weight="bold")
    draw_table(axes[0], per_class, "Klasyfikacja: metryki per class", [0.46, 0.13, 0.13, 0.13, 0.15], fontsize=8)
    draw_table(axes[1], cluster_table, "K-Means: skrócone profile klastrów", [0.10, 0.12, 0.12, 0.19, 0.14, 0.14, 0.19], fontsize=8)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def image_sheet(pdf, title, stems):
    paths = [FIGURES / f"{stem}.png" for stem in stems]
    paths = [path for path in paths if path.exists()]
    if not paths:
        return
    fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    if len(paths) == 3:
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    elif len(paths) == 2:
        axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[:, 1])]
    else:
        axes = [fig.add_subplot(grid[:, :])]
    fig.suptitle(title, fontsize=16, weight="bold")
    for index, (ax, path) in enumerate(zip(axes, paths)):
        ax.imshow(plt.imread(path))
        ax.axis("off")
        width = 72 if len(paths) == 3 and index == 2 else 46
        caption = "\n".join(wrap(CAPTIONS[path.stem], width=width))
        ax.text(0.5, -0.03, caption, transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    pdf.savefig(fig)
    plt.close(fig)


def build_combined_overview():
    regression = pd.read_csv(REPORTS / "regression_model_comparison.csv")
    classification = pd.read_csv(REPORTS / "classification_model_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(regression["model"], regression["test_MAE_USD"], color="#4E79A7")
    axes[0].set_title("Regresja — test MAE [USD]")
    axes[0].set_xlabel("mniej = lepiej")
    axes[1].barh(classification["model"], classification["test_macro_F1"], color="#59A14F")
    axes[1].set_title("Klasyfikacja — test Macro F1")
    axes[1].set_xlabel("więcej = lepiej")
    axes[1].set_xlim(0, 1)
    fig.suptitle("Zbiorcze porównanie modeli", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_00_combined_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    build_combined_overview()
    with PdfPages(OUTPUT) as pdf:
        text_page(pdf, "Data Professional Salary Survey Analysis", [
            "Raport z EDA, przygotowania danych bez leakage, regresji, klasyfikacji i klasteryzacji.",
            "Preprocessing pozostaje wewnątrz Pipeline; OtherDatabases jest kodowane jako multi-label; Counter oraz pola identyfikacyjne są wykluczone; techniczna klasa Not Asked nie uczestniczy w klasyfikacji.",
            "Wynagrodzenia są nominalne. Regresja wykorzystuje log1p celu, ale MAE i RMSE raportuje w USD. Cały raport mieści się w wymaganym zakresie 10–20 stron.",
        ])
        text_page(pdf, "Najważniejsze wyniki i zakres zastosowania", [
            "Regresja: XGBoost osiąga test MAE około 19 tys. USD. Wynik jest użyteczny jako orientacyjna estymacja na poziomie grup, lecz nie usuwa niepewności indywidualnej.",
            "Klasyfikacja: test Macro F1 i balanced accuracy wynoszą około 0,25. Model pozwala na eksploracyjne rozróżnianie planów zawodowych, lecz jego jakość jest zbyt niska do decyzji kadrowych lub automatycznej klasyfikacji produkcyjnej.",
            "Klasteryzacja: K-Means daje słabą, ale opisywalną segmentację. DBSCAN nie potwierdził stabilnych klastrów, dlatego wyniki mają charakter badawczy, a nie operacyjny.",
            "Walidacja po tuningu jest non-nested CV: 3-fold CV wybiera hiperparametry, a 5-fold CV ponownie ocenia ustaloną konfigurację na tym samym train. Może to lekko zawyżać CV; niezależny test pozostaje oceną końcową.",
        ])
        summary_tables_page(pdf)
        detail_tables_page(pdf)
        for title, stems in FIGURE_GROUPS:
            image_sheet(pdf, title, stems)
        text_page(pdf, "Ograniczenia i wnioski końcowe", [
            "Dane są samoopisowe i mogą nie reprezentować całego rynku; analiza nie uzasadnia wniosków przyczynowych. Wynagrodzenia nie są skorygowane o inflację, kursy ani PPP.",
            "Losowy split miesza lata, dlatego regresja zawiera także temporal holdout. Test nie uczestniczy w dopasowaniu preprocessingu, wyborze cech ani hiperparametrów.",
            "Niska jakość klasyfikacji i słaby silhouette wykluczają zastosowania produkcyjne bez nowych cech, danych zewnętrznych oraz prospektywnej walidacji.",
            "Najbardziej wiarygodnym rezultatem jest pipeline regresyjny; klasyfikację i klasteryzację należy traktować jako analizy eksploracyjne.",
        ])
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
