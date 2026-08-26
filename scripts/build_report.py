"""Build a PDF report from generated metrics and every saved project figure."""

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
    "eda_01_missingness_by_employment": "Braki zależą od statusu zatrudnienia; imputacja i wskaźniki braków muszą być uczone na train.",
    "eda_02_all_numeric_distributions": "Skośność, długie ogony i stała Counter uzasadniają transformację celu oraz reguły jakości.",
    "eda_03_salary_hist_kde": "SalaryUSD jest silnie prawostronnie skośne; wysokie wartości wpływają na błędy w USD.",
    "eda_04_salary_log_hist": "log1p stabilizuje rozkład celu; wyniki biznesowe są następnie przeliczane z powrotem na USD.",
    "eda_05_anomalies_boxplots": "Odstające wynagrodzenia są analizowane, a staż powyżej 60 lat jest traktowany jako błąd jakości.",
    "eda_06_salary_top_countries": "Kraj silnie różnicuje medianę i rozrzut wynagrodzeń i jest istotną cechą modelu.",
    "eda_07_salary_top_jobs": "Stanowisko wpływa na wynagrodzenie, lecz nakładanie grup wymaga analizy wielowymiarowej.",
    "eda_08_salary_by_year": "Zmiana mediany w czasie uzasadnia zachowanie Survey Year i dodatkowy temporal split.",
    "eda_09_experience_salary_nonlinear": "Zależność stażu i wynagrodzenia jest dodatnia, lecz nieliniowa i silnie rozproszona.",
    "eda_10_salary_experience_level": "Mediana rośnie z doświadczeniem, ale grupy nakładają się; potrzebne są także pozostałe cechy.",
    "eda_11_career_by_employment": "Struktura planów zależy od zatrudnienia; Not Asked pozostaje artefaktem ankiety 2017.",
    "eda_12_career_by_experience": "Plany różnią się według doświadczenia, a nierównowaga klas wymaga Macro F1.",
    "eda_13_other_databases_frequency": "OtherDatabases to multi-label; rozbijanie list redukuje tysiące sztucznych kombinacji.",
    "eda_14_category_frequencies": "Nierówne częstości uzasadniają train-fitted grouping rzadkich kategorii.",
    "eda_15_spearman_correlations": "Słabe korelacje parami nie wykluczają nieliniowości i interakcji wykorzystywanych przez drzewa.",
    "model_01_regression_comparison": "Train, CV i test pokazują jakość generalizacji; MAE jest raportowane w nominalnych USD.",
    "model_02_regression_importance": "Permutation importance mierzy wpływ predykcyjny całych surowych kolumn, nie przyczynowość.",
    "model_03_classification_comparison": "Macro F1 porównuje modele bez sztucznej klasy Not Asked i nadaje klasom równą wagę.",
    "model_04_classification_confusion_matrix": "Znormalizowana macierz pokazuje recall oraz pary klas najczęściej ze sobą mylone.",
    "model_05_classification_importance": "Ważności opisują predykcję planów zawodowych po usunięciu artefaktu 2017.",
    "cluster_01_svd_explained_variance": "Dwa komponenty nie zachowują całej struktury; służą wyłącznie do wizualizacji.",
    "cluster_02_k_selection": "Elbow i silhouette zastępują arbitralny wybór liczby klastrów.",
    "cluster_03_kmeans_2d": "Wykres jest rzutem 2D, podczas gdy klasteryzacja wykorzystuje 20 komponentów.",
    "cluster_04_profiles": "Profile według kraju, roli, stażu i pensji nadają segmentom interpretację biznesową.",
    "model_00_combined_overview": "Panele zestawiają modele w ramach właściwych metryk; MAE i Macro F1 nie są porównywane bezpośrednio.",
}


def text_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=20, weight="bold", va="top")
    y = 0.88
    for paragraph in paragraphs:
        lines = wrap(paragraph, width=95)
        fig.text(0.08, y, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.45)
        y -= 0.034 * len(lines) + 0.035
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(pdf, csv_path, title):
    if not csv_path.exists():
        return
    data = pd.read_csv(csv_path).round(3)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=20)
    table = ax.table(cellText=data.values, colLabels=data.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.45)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_combined_overview():
    regression_path = REPORTS / "regression_model_comparison.csv"
    classification_path = REPORTS / "classification_model_comparison.csv"
    if not regression_path.exists() or not classification_path.exists():
        return
    regression = pd.read_csv(regression_path)
    classification = pd.read_csv(classification_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(regression["model"], regression["test_MAE_USD"], color="#4E79A7")
    axes[0].set_title("Regresja — test MAE [USD], mniej = lepiej")
    axes[0].set_xlabel("MAE [USD]")
    axes[1].barh(classification["model"], classification["test_macro_F1"], color="#59A14F")
    axes[1].set_title("Klasyfikacja — test Macro F1, więcej = lepiej")
    axes[1].set_xlabel("Macro F1")
    axes[1].set_xlim(0, 1)
    fig.suptitle("Zbiorcze porównanie modeli", fontsize=16, weight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "model_00_combined_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


build_combined_overview()


with PdfPages(OUTPUT) as pdf:
    text_page(pdf, "Data Professional Salary Survey Analysis", [
        "Raport z eksploracyjnej analizy danych, przygotowania danych bez leakage, regresji, klasyfikacji i klasteryzacji.",
        "Najważniejsze korekty metodologiczne: preprocessing wewnątrz Pipeline po train/test split; multi-label encoding pola OtherDatabases; usunięcie stałej Counter; zachowanie Survey Year; wykluczenie technicznej klasy Not Asked z klasyfikacji.",
        "Metryki regresji w USD są liczone po odwrotnej transformacji expm1. R2_log dotyczy przestrzeni log1p(SalaryUSD). Kwoty są nominalne i nie są skorygowane o inflację ani parytet siły nabywczej.",
    ])
    table_page(pdf, REPORTS / "regression_model_comparison.csv", "Regresja: train vs CV vs test")
    table_page(pdf, REPORTS / "classification_model_comparison.csv", "Klasyfikacja: train vs CV vs test")
    table_page(pdf, REPORTS / "classification_per_class_metrics.csv", "Klasyfikacja: metryki per class")
    table_page(pdf, REPORTS / "cluster_profiles.csv", "Profile klastrów")

    for image_path in sorted(FIGURES.glob("*.png")):
        image = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.set_position([0.05, 0.16, 0.90, 0.76])
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(image_path.stem.replace("_", " "), fontsize=12, pad=12)
        caption = CAPTIONS.get(image_path.stem, "Wykres uzupełnia analizę i jest interpretowany szczegółowo w odpowiednim notebooku.")
        fig.text(0.07, 0.07, "Interpretacja: " + "\n".join(wrap(caption, width=120)),
                 fontsize=9.5, va="bottom")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    text_page(pdf, "Ograniczenia i zagrożenia trafności", [
        "Dane są samoopisowe i pochodzą z próby ankietowej, która może nie reprezentować całego rynku. Analiza nie uzasadnia wniosków przyczynowych.",
        "Losowy split miesza lata 2017–2026; dlatego regresja zawiera dodatkowy temporal split. Survey Year pozostaje cechą, jednak wynagrodzenia są nominalne i bez korekty inflacyjnej.",
        "Not Asked pochodzi wyłącznie z 2017 roku i nie jest prawdziwą kategorią planów zawodowych. Wyniki głównej klasyfikacji dotyczą wyłącznie respondentów, którym zadano pytanie.",
        "Dobór reguł jakości, reprezentacji multi-label, hiperparametrów SVD oraz algorytmów wpływa na rezultaty. DBSCAN z jednym klastrem należy traktować jako wynik negatywny.",
        "Test nie uczestniczy w dopasowaniu preprocessingu ani wyborze hiperparametrów; ostateczne metryki testowe nie powinny być dalej optymalizowane.",
    ])

print(f"Saved {OUTPUT}")
