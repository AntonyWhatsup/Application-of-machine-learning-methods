"""Generate the reviewed, reproducible notebooks used in the project.

Keeping notebook sources here makes large JSON notebooks reviewable and repeatable.
"""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(name: str, cells):
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
    )
    nbf.write(notebook, NOTEBOOKS / name)


COMMON_SETUP = r'''
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
'''


eda_cells = [
    md('''
    # 1. Eksploracyjna analiza danych (EDA)

    Celem notebooka jest pełna diagnoza jakości danych, rozkładów, braków, anomalii,
    wysokiej kardynalności oraz zależności istotnych dla regresji i klasyfikacji.
    Każda wizualizacja zawiera krótką interpretację i konsekwencje dla modelowania.
    ''') ,
    code(COMMON_SETUP + r'''
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

from salary_survey.config import DATA_RAW_PATH, FIGURES_DIR
from salary_survey.data import add_domain_features, load_raw_data

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.labelsize": 10})

def finish_plot(filename, note):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180, bbox_inches="tight")
    plt.show()
    display(Markdown(f"**Interpretacja.** {note}"))

df = load_raw_data(DATA_RAW_PATH)
df_eda = add_domain_features(df)
print(f"Liczba obserwacji: {len(df):,}; liczba zmiennych: {df.shape[1]}")
display(df.head(3))
'''),
    md('''
    ## Struktura i typy wszystkich zmiennych

    Tabela obejmuje typ techniczny, liczbę wartości unikalnych, braki i przykładowe wartości.
    Pozwala odróżnić zmienne liczbowe, kategoryczne, czasowe, stałe i pola tekstowe.
    '''),
    code(r'''
structure = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "missing": df.isna().sum(),
    "missing_pct": (100 * df.isna().mean()).round(2),
    "n_unique": df.nunique(dropna=False),
    "examples": [", ".join(map(str, df[c].dropna().astype(str).unique()[:3])) for c in df.columns],
})
display(structure)
display(Markdown(
    "**Wniosek.** `Counter` jest stałą i nie wnosi informacji. `Timestamp` oraz `PostalCode` "
    "mają charakter identyfikacyjny. `OtherDatabases` jest polem wielokrotnego wyboru, a nie "
    "pojedynczą zmienną kategoryczną. Te cechy wymagają odmiennej obsługi."
))
'''),
    md('''## Braki danych i ich zależność od `EmploymentStatus`'''),
    code(r'''
missing_table = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_pct": (100 * df.isna().mean()).round(2),
}).query("missing_count > 0").sort_values("missing_pct", ascending=False)
display(missing_table)

missing_by_status = (
    df.assign(**{f"MISS__{c}": df[c].isna() for c in missing_table.index})
      .groupby("EmploymentStatus")[[f"MISS__{c}" for c in missing_table.index]]
      .mean().mul(100)
)
missing_by_status.columns = [c.replace("MISS__", "") for c in missing_by_status.columns]
display(missing_by_status.round(1))

plt.figure(figsize=(12, max(4, 0.55 * len(missing_by_status))))
sns.heatmap(missing_by_status, annot=True, fmt=".0f", cmap="YlOrRd", vmin=0, vmax=100)
plt.title("Odsetek braków danych według statusu zatrudnienia")
plt.xlabel("Zmienna")
plt.ylabel("EmploymentStatus")
finish_plot(
    "eda_01_missingness_by_employment.png",
    "Braki nie są rozłożone losowo: ich udział zmienia się między statusami zatrudnienia. "
    "Dlatego wskaźniki braków mogą nieść informację, a imputacja musi być uczona wyłącznie na zbiorze treningowym."
)
'''),
    md('''## Rozkłady wszystkich zmiennych liczbowych'''),
    code(r'''
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
ncols = 3
nrows = int(np.ceil(len(numeric_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
for ax, column in zip(np.ravel(axes), numeric_cols):
    sns.histplot(df[column].dropna(), bins=35, kde=df[column].nunique() > 5, ax=ax)
    ax.set_title(column)
for ax in np.ravel(axes)[len(numeric_cols):]:
    ax.remove()
fig.suptitle("Rozkłady wszystkich zmiennych liczbowych", y=1.01, fontsize=15)
finish_plot(
    "eda_02_all_numeric_distributions.png",
    "Rozkłady są silnie zróżnicowane: wynagrodzenie jest prawostronnie skośne, zmienne stażowe "
    "mają długie ogony, `MonthsUnemployed` jest niemal całkowicie puste, a `Counter` jest stałą. "
    "Wskazuje to na transformację logarytmiczną celu, czyszczenie stażu oraz usunięcie `Counter`."
)
'''),
    md('''## `SalaryUSD` w skali zwykłej i logarytmicznej'''),
    code(r'''
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["SalaryUSD"], bins=60, kde=True, ax=ax, color="#2878B5")
ax.set_title("Rozkład SalaryUSD — skala nominalna")
ax.set_xlabel("Roczne wynagrodzenie [USD]")
finish_plot(
    "eda_03_salary_hist_kde.png",
    "Rozkład wynagrodzeń jest silnie prawostronnie skośny; nieliczne bardzo wysokie wartości dominują skalę. "
    "MAE w USD pozostaje czytelne biznesowo, ale modelowanie logarytmu stabilizuje wariancję."
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(np.log1p(df["SalaryUSD"]), bins=60, kde=True, ax=ax, color="#F28E2B")
ax.set_title("Rozkład log1p(SalaryUSD)")
ax.set_xlabel("log1p(roczne wynagrodzenie [USD])")
finish_plot(
    "eda_04_salary_log_hist.png",
    "Po transformacji log1p rozkład jest znacznie bardziej symetryczny. Regresja będzie uczona w log-space, "
    "natomiast MAE i RMSE zostaną również raportowane po transformacji odwrotnej w USD."
)
'''),
    md('''## Analiza anomalii: wynagrodzenie i staż'''),
    code(r'''
def iqr_summary(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return pd.Series({"Q1": q1, "Q3": q3, "IQR": iqr, "lower": low, "upper": high,
                      "outside_IQR": ((series < low) | (series > high)).sum()})

anomaly_table = pd.DataFrame({
    c: iqr_summary(df[c].dropna())
    for c in ["SalaryUSD", "YearsWithThisDatabase", "YearsWithThisTypeOfJob"]
}).T
anomaly_table["above_60"] = [np.nan,
    (df["YearsWithThisDatabase"] > 60).sum(),
    (df["YearsWithThisTypeOfJob"] > 60).sum()]
display(anomaly_table)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.boxplot(x=df["SalaryUSD"], ax=axes[0], color="#59A14F")
sns.boxplot(x=df["YearsWithThisDatabase"], ax=axes[1], color="#E15759")
sns.boxplot(x=df["YearsWithThisTypeOfJob"], ax=axes[2], color="#B07AA1")
axes[0].set_title("SalaryUSD")
axes[1].set_title("Staż z bazą danych")
axes[2].set_title("Staż w typie pracy")
finish_plot(
    "eda_05_anomalies_boxplots.png",
    "Wartości odstające wynagrodzenia nie muszą być błędami, dlatego są analizowane oddzielnie od reguł jakości. "
    "Staż powyżej 60 lat uznajemy za niewiarygodny i zamieniamy na brak przed imputacją w pipeline."
)
'''),
    md('''## Wynagrodzenie według kraju, stanowiska i roku'''),
    code(r'''
top_countries = df["Country"].value_counts().head(10).index
country_medians = (df[df["Country"].isin(top_countries)]
                   .groupby("Country")["SalaryUSD"].agg(["count", "median"])
                   .sort_values("median", ascending=False))
display(country_medians)
order = country_medians.index
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df["Country"].isin(top_countries)], x="SalaryUSD", y="Country", order=order,
            showfliers=False)
plt.title("Wynagrodzenie w 10 najliczniejszych krajach")
plt.xlabel("SalaryUSD (bez punktów poza wąsami dla czytelności)")
finish_plot(
    "eda_06_salary_top_countries.png",
    "Mediany różnią się wyraźnie między krajami, więc kraj jest ważnym predyktorem. "
    "Duże różnice liczebności i rozrzutu wymagają `handle_unknown` oraz łączenia rzadkich kategorii na train."
)

top_jobs = df["JobTitle"].value_counts().head(10).index
job_medians = (df[df["JobTitle"].isin(top_jobs)]
               .groupby("JobTitle")["SalaryUSD"].agg(["count", "median"])
               .sort_values("median", ascending=False))
display(job_medians)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df["JobTitle"].isin(top_jobs)], x="SalaryUSD", y="JobTitle",
            order=job_medians.index, showfliers=False)
plt.title("Wynagrodzenie na 10 najczęstszych stanowiskach")
plt.xlabel("SalaryUSD (bez punktów poza wąsami dla czytelności)")
finish_plot(
    "eda_07_salary_top_jobs.png",
    "Stanowisko różnicuje medianę i rozrzut wynagrodzeń, ale nakładanie się pudełek pokazuje, "
    "że sama nazwa roli nie wystarcza — potrzebny jest model wielowymiarowy."
)

salary_by_year = df.groupby("Survey Year")["SalaryUSD"].agg(["count", "median", "mean"]).reset_index()
display(salary_by_year)
plt.figure(figsize=(10, 5))
sns.lineplot(data=salary_by_year, x="Survey Year", y="median", marker="o", linewidth=2.5)
plt.title("Mediana nominalnego SalaryUSD według roku ankiety")
plt.ylabel("Mediana SalaryUSD")
finish_plot(
    "eda_08_salary_by_year.png",
    "Mediana zmienia się w czasie, dlatego `Survey Year` pozostaje cechą i wykonujemy dodatkowy temporal split. "
    "Kwoty są nominalne w USD i nie zostały skorygowane o inflację ani zmianę kursów."
)
'''),
    md('''## Wynagrodzenie a doświadczenie — zależność liniowa i nieliniowa'''),
    code(r'''
valid_scatter = df.query("SalaryUSD >= 5000 and SalaryUSD <= 500000 and YearsWithThisTypeOfJob <= 60").copy()
sample = valid_scatter.sample(min(5000, len(valid_scatter)), random_state=42)
plt.figure(figsize=(10, 6))
sns.regplot(data=sample, x="YearsWithThisTypeOfJob", y="SalaryUSD", order=2,
            scatter_kws={"alpha": 0.15, "s": 18}, line_kws={"color": "#D62728", "linewidth": 3})
plt.title("Staż w typie pracy a SalaryUSD — regresja wielomianowa")
plt.ylim(0, 300000)
finish_plot(
    "eda_09_experience_salary_nonlinear.png",
    "Trend jest dodatni, ale nieliniowy i obarczony dużym rozrzutem. Uzasadnia to cechę `Years_Squared` "
    "oraz porównanie modelu liniowego z modelami drzewiastymi."
)

experience_salary = df_eda.groupby("Experience_Level", observed=False)["SalaryUSD"].agg(["count", "median"])
display(experience_salary)
plt.figure(figsize=(9, 5))
sns.boxplot(data=df_eda, x="Experience_Level", y="SalaryUSD",
            order=["Junior", "Mid", "Senior", "Unknown"], showfliers=False)
plt.title("SalaryUSD według poziomu doświadczenia")
plt.ylim(0, 300000)
finish_plot(
    "eda_10_salary_experience_level.png",
    "Mediana wynagrodzenia rośnie wraz z poziomem doświadczenia, ale grupy nadal mocno się nakładają. "
    "Kategoria doświadczenia jest użyteczna jako uzupełnienie, nie zamiennik wartości liczbowej stażu."
)
'''),
    md('''## `CareerPlansThisYear` według zatrudnienia i doświadczenia'''),
    code(r'''
career_by_status = pd.crosstab(df["EmploymentStatus"], df["CareerPlansThisYear"], normalize="index")
display(career_by_status.round(3))
career_by_status.plot(kind="barh", stacked=True, figsize=(13, 7), colormap="tab20")
plt.title("Struktura planów zawodowych według statusu zatrudnienia")
plt.xlabel("Udział w grupie")
plt.ylabel("EmploymentStatus")
plt.legend(title="CareerPlansThisYear", bbox_to_anchor=(1.02, 1), loc="upper left")
finish_plot(
    "eda_11_career_by_employment.png",
    "Proporcje planów różnią się między statusami zatrudnienia, więc `EmploymentStatus` jest uzasadnioną cechą. "
    "Jednocześnie klasa `Not Asked` wymaga osobnego potraktowania jako artefakt ankiety z 2017 roku."
)

career_by_exp = pd.crosstab(df_eda["Experience_Level"], df_eda["CareerPlansThisYear"], normalize="index")
display(career_by_exp.round(3))
career_by_exp.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
plt.title("Struktura planów zawodowych według poziomu doświadczenia")
plt.ylabel("Udział w grupie")
plt.xlabel("Experience_Level")
plt.xticks(rotation=0)
plt.legend(title="CareerPlansThisYear", bbox_to_anchor=(1.02, 1), loc="upper left")
finish_plot(
    "eda_12_career_by_experience.png",
    "Rozkład celu zmienia się wraz z doświadczeniem. Zmienna może pomagać w klasyfikacji, "
    "ale jakość należy oceniać przez macro F1 i balanced accuracy ze względu na nierównowagę klas."
)
'''),
    md('''## Wysoka kardynalność `OtherDatabases` i częstości kategorii'''),
    code(r'''
categorical_cardinality = (df.select_dtypes(include=["object", "category"])
                           .nunique().sort_values(ascending=False).rename("n_unique").to_frame())
display(categorical_cardinality)

tokens = (df["OtherDatabases"].dropna().astype(str).str.split(",").explode().str.strip())
token_counts = tokens.value_counts().head(20)
raw_combinations = df["OtherDatabases"].nunique(dropna=True)
display(pd.DataFrame({"raw_unique_combinations": [raw_combinations],
                      "unique_individual_databases": [tokens.nunique()]}))

plt.figure(figsize=(11, 7))
sns.barplot(x=token_counts.values, y=token_counts.index, color="#4E79A7")
plt.title("20 najczęściej wskazywanych baz w polu OtherDatabases")
plt.xlabel("Liczba wskazań")
plt.ylabel("")
finish_plot(
    "eda_13_other_databases_frequency.png",
    f"Pole ma {raw_combinations:,} unikalnych kombinacji, lecz składa się z powtarzalnych nazw baz. "
    "One-hot całych ciągów tworzył tysiące sztucznych kategorii; pipeline rozdzieli listę po przecinku "
    "i zachowa tylko częste etykiety wyznaczone na train."
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, column, limit in [(axes[0], "PrimaryDatabase", 15), (axes[1], "JobTitle", 15)]:
    counts = df[column].value_counts().head(limit)
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(f"Najczęstsze kategorie: {column}")
    ax.set_xlabel("Liczba odpowiedzi")
    ax.set_ylabel("")
finish_plot(
    "eda_14_category_frequencies.png",
    "Częstości kategorii są nierówne. Rzadkie poziomy zwiększają wariancję estymacji, "
    "dlatego encoder grupuje je parametrem `min_frequency` dopasowanym wyłącznie na train."
)
'''),
    md('''## Korelacje i zależności nieliniowe'''),
    code(r'''
corr_columns = [c for c in numeric_cols if c != "Counter"]
corr = df[corr_columns].corr(method="spearman")
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
plt.title("Korelacje rang Spearmana — bez stałej Counter")
finish_plot(
    "eda_15_spearman_correlations.png",
    "Korelacje liniowo-monotoniczne z wynagrodzeniem są umiarkowane lub słabe. "
    "Nie oznacza to braku predykcji: wcześniejszy wykres ujawnia nieliniowość oraz interakcje z krajem i stanowiskiem, "
    "co uzasadnia modele drzewiaste."
)
'''),
    md('''
    ## Wnioski z EDA

    1. `SalaryUSD` jest prawostronnie skośne — celem regresji będzie `log1p(SalaryUSD)`, a wyniki będą raportowane również w USD.
    2. `Counter` jest stałą; `Timestamp` i `PostalCode` nie będą cechami modelu.
    3. Staż powyżej 60 lat zostanie zamieniony na brak i imputowany wewnątrz pipeline.
    4. `OtherDatabases` jest polem multi-label i nie może być kodowane jako 2541 osobnych kombinacji.
    5. `Survey Year` pozostaje cechą; potrzebna jest dodatkowa walidacja czasowa, a wartości są nominalne i bez korekty inflacyjnej.
    6. `Not Asked` pochodzi wyłącznie z 2017 roku i zostanie wykluczone z głównej klasyfikacji.
    7. Preprocessing statystyczny musi być uczony po podziale danych, wyłącznie na train/fold.
    ''')
]


prep_cells = [
    md('''
    # 2. Przygotowanie danych i inżynieria cech bez leakage

    Notebook wykonuje wyłącznie deterministyczne reguły domenowe i zapisuje dane w formie niezakodowanej.
    Imputacja, grupowanie rzadkich kategorii, multi-label encoding i skalowanie znajdują się w
    `Pipeline`/`ColumnTransformer` i są dopasowywane dopiero po podziale train/test.
    '''),
    code(COMMON_SETUP + r'''
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split

from salary_survey.config import DATA_PROCESSED_PATH, DATA_RAW_PATH, RANDOM_STATE
from salary_survey.data import clean_domain_errors, filter_salary_scope, load_raw_data
from salary_survey.pipelines import build_preprocessor

raw = load_raw_data(DATA_RAW_PATH)
print("Raw shape:", raw.shape)
'''),
    md('''## Reguły jakości danych i cechy domenowe'''),
    code(r'''
quality_before = pd.DataFrame({
    "rule": ["YearsWithThisDatabase > 60", "YearsWithThisTypeOfJob > 60", "Counter constant"],
    "affected_rows": [
        (raw["YearsWithThisDatabase"] > 60).sum(),
        (raw["YearsWithThisTypeOfJob"] > 60).sum(),
        int(raw["Counter"].nunique(dropna=False) == 1),
    ],
})
display(quality_before)

clean = clean_domain_errors(raw)
salary_modeling = filter_salary_scope(clean, lower=5_000, upper=500_000)
salary_modeling["SalaryUSD_Log"] = np.log1p(salary_modeling["SalaryUSD"])

feature_audit = pd.DataFrame({
    "feature": ["Experience_Level", "Specialization_Ratio", "Years_Squared"],
    "purpose": [
        "nieliniowa segmentacja stażu: Junior/Mid/Senior",
        "udział doświadczenia z bazą w stażu zawodowym",
        "krzywizna zależności stażu i wynagrodzenia",
    ],
})
display(feature_audit)
print("Domain-clean shape (saved for all tasks):", clean.shape)
print("Salary-scope shape (regression only):", salary_modeling.shape)
print("Counter present:", "Counter" in clean.columns)
print("Invalid database experience remaining:", (clean["YearsWithThisDatabase"] > 60).sum())
print("Invalid job experience remaining:", (clean["YearsWithThisTypeOfJob"] > 60).sum())
'''),
    md('''
    **Uzasadnienie.** Granica 60 lat jest jawną regułą jakości, a nie statystyką wyliczoną z danych.
    Wartości powyżej granicy zamieniamy na `NaN`; mediany do imputacji pozna dopiero pipeline na train.
    Zakres wynagrodzenia 5 000–500 000 USD definiuje analizowaną populację i usuwa oczywiste błędy jednostek.
    '''),
    md('''## Najpierw split, potem dopasowanie preprocessingu'''),
    code(r'''
drop_features = ["SalaryUSD", "SalaryUSD_Log", "CareerPlansThisYear", "Timestamp", "PostalCode"]
X = salary_modeling.drop(columns=drop_features, errors="ignore")
y = salary_modeling["SalaryUSD_Log"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
preprocessor = build_preprocessor(X_train)
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)

split_table = pd.DataFrame({
    "split": ["train", "test"],
    "rows": [len(X_train), len(X_test)],
    "raw_features": [X_train.shape[1], X_test.shape[1]],
    "transformed_features": [X_train_ready.shape[1], X_test_ready.shape[1]],
})
display(split_table)
db_encoder = preprocessor.named_transformers_["other_databases"]
display(pd.DataFrame({"OtherDatabases labels learned on train": db_encoder.classes_}))
display(Markdown(
    "**Kontrola leakage.** `fit_transform` wykonano tylko dla `X_train`; dla `X_test` użyto wyłącznie `transform`. "
    "To samo zachowanie zostanie automatycznie powtórzone w każdym foldzie CV, ponieważ preprocessor jest częścią Pipeline."
))
'''),
    md('''## Zapis danych po regułach domenowych, przed statystycznym preprocessingiem'''),
    code(r'''
DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(DATA_PROCESSED_PATH, index=False)
print(f"Saved: {DATA_PROCESSED_PATH}")
print(f"Size: {DATA_PROCESSED_PATH.stat().st_size / 1024**2:.2f} MB")
'''),
    md('''
    ## Podsumowanie metodologii

    - `Survey Year` pozostaje w danych modelowych.
    - `Counter` usunięto, a identyfikacyjne `Timestamp` i `PostalCode` są wyłączane z cech modelu.
    - `OtherDatabases` jest rozbijane po przecinku i kodowane jako multi-label.
    - imputer, rare-category handling, encoder i scaler są elementami pipeline;
    - ani test, ani fold walidacyjny nie wpływają na parametry preprocessingu.
    ''')
]


write_notebook("01_eda.ipynb", eda_cells)
write_notebook("02_data_prep_and_feat_eng.ipynb", prep_cells)
print("Generated EDA and preparation notebooks")


regression_cells = [
    md('''
    # 3. Modelowanie regresyjne

    Cel: prognoza `log1p(SalaryUSD)` z raportowaniem błędów również w nominalnych USD.
    Pipeline obejmuje cały preprocessing, więc każdy fold CV uczy imputer, encoder, grupowanie
    rzadkich kategorii, multi-label encoding i scaler wyłącznie na części treningowej.
    ''') ,
    code(COMMON_SETUP + r'''
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from salary_survey.config import DATA_PROCESSED_PATH, FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from salary_survey.pipelines import build_preprocessor

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
df = pd.read_csv(DATA_PROCESSED_PATH)
df = df.loc[df["SalaryUSD"].between(5_000, 500_000)].copy()
df["SalaryUSD_Log"] = np.log1p(df["SalaryUSD"])

feature_exclusions = ["SalaryUSD", "SalaryUSD_Log", "CareerPlansThisYear", "Timestamp", "PostalCode"]
X = df.drop(columns=feature_exclusions, errors="ignore")
y = df["SalaryUSD_Log"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Survey Year retained:", "Survey Year" in X.columns)
'''),
    md('''## Metryki i walidacja'''),
    code(r'''
def mae_usd(y_true_log, y_pred_log):
    return mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))

mae_usd_scorer = make_scorer(mae_usd, greater_is_better=False)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"mae_usd": mae_usd_scorer, "r2_log": "r2"}

def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor(X_train)), ("model", model)])

def evaluate_model(name, estimator):
    scores = cross_validate(
        estimator, X_train, y_train, cv=cv, scoring=scoring,
        return_train_score=True, n_jobs=1,
    )
    estimator.fit(X_train, y_train)
    train_pred = estimator.predict(X_train)
    test_pred = estimator.predict(X_test)
    row = {
        "model": name,
        "train_MAE_USD": mae_usd(y_train, train_pred),
        "CV_MAE_USD_mean": -scores["test_mae_usd"].mean(),
        "CV_MAE_USD_std": scores["test_mae_usd"].std(),
        "test_MAE_USD": mae_usd(y_test, test_pred),
        "test_RMSE_USD": root_mean_squared_error(np.expm1(y_test), np.expm1(test_pred)),
        "train_R2_log": r2_score(y_train, train_pred),
        "CV_R2_log_mean": scores["test_r2_log"].mean(),
        "CV_R2_log_std": scores["test_r2_log"].std(),
        "test_R2_log": r2_score(y_test, test_pred),
    }
    return estimator, row

models = {
    "Ridge default": make_pipeline(Ridge(alpha=1.0, solver="lsqr")),
    "Random Forest default": make_pipeline(RandomForestRegressor(
        n_estimators=150, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    )),
    "XGBoost default": make_pipeline(XGBRegressor(
        n_estimators=180, max_depth=6, learning_rate=0.1, subsample=1.0,
        colsample_bytree=1.0, random_state=RANDOM_STATE, n_jobs=-1,
    )),
}
results = []
fitted = {}
for name, estimator in models.items():
    print("Evaluating", name)
    fitted[name], row = evaluate_model(name, estimator)
    results.append(row)
display(pd.DataFrame(results).round(4))
'''),
    md('''
    **Interpretacja metryk.** `MAE_USD` i `RMSE_USD` są liczone po `expm1`, więc mają jednostkę USD.
    `R2_log` jest liczony w przestrzeni logarytmicznej używanej podczas uczenia. Standardowe odchylenie CV
    pokazuje stabilność między foldami; różnica train–CV wskazuje skalę przeuczenia.
    '''),
    md('''## Strojenie XGBoost i porównanie default vs tuned'''),
    code(r'''
xgb_pipeline = make_pipeline(XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1))
param_distributions = {
    "model__n_estimators": [120, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.75, 0.9, 1.0],
    "model__colsample_bytree": [0.75, 0.9, 1.0],
}
xgb_search = RandomizedSearchCV(
    xgb_pipeline, param_distributions=param_distributions, n_iter=8,
    scoring=mae_usd_scorer, cv=3, random_state=RANDOM_STATE, n_jobs=1,
    return_train_score=True, verbose=1,
)
xgb_search.fit(X_train, y_train)
print("Best parameters:", xgb_search.best_params_)
best_xgb, tuned_row = evaluate_model("XGBoost tuned", xgb_search.best_estimator_)
results.append(tuned_row)
fitted["XGBoost tuned"] = best_xgb

regression_results = pd.DataFrame(results).set_index("model").sort_values("CV_MAE_USD_mean")
best_model_name = regression_results.index[0]
best_regressor = fitted[best_model_name]
display(regression_results.round(4))
print("Selected by CV MAE:", best_model_name)
regression_results.to_csv(PROJECT_ROOT / "reports" / "regression_model_comparison.csv")
joblib.dump(best_regressor, MODELS_DIR / "regression_best_pipeline.joblib")
'''),
    md('''## Porównanie modeli: train, CV i test'''),
    code(r'''
plot_data = (regression_results.reset_index()
             .melt(id_vars="model", value_vars=["train_MAE_USD", "CV_MAE_USD_mean", "test_MAE_USD"],
                   var_name="split", value_name="MAE_USD"))
plt.figure(figsize=(12, 6))
sns.barplot(data=plot_data, x="model", y="MAE_USD", hue="split")
plt.title("Porównanie modeli regresyjnych — MAE w nominalnych USD")
plt.xlabel("")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_01_regression_comparison.png", dpi=180, bbox_inches="tight")
plt.show()
display(Markdown(
    "**Interpretacja.** Najlepszy model wybieramy według CV MAE, a test służy tylko do końcowej oceny. "
    "Zestawienie default–tuned pokazuje rzeczywistą wartość strojenia; duża luka train–CV oznacza przeuczenie."
))
'''),
    md('''## Interpretacja najlepszego modelu'''),
    code(r'''
sample_idx = X_test.sample(min(1200, len(X_test)), random_state=RANDOM_STATE).index
importance = permutation_importance(
    best_regressor, X_test.loc[sample_idx], y_test.loc[sample_idx],
    scoring=mae_usd_scorer, n_repeats=4, random_state=RANDOM_STATE, n_jobs=1,
)
importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
}).sort_values("importance_mean", ascending=False).head(15)
display(importance_df)
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="importance_mean", y="feature", color="#4E79A7")
plt.title("Permutation importance — najlepszy model regresyjny")
plt.xlabel("Spadek jakości po permutacji (większy = ważniejsza cecha)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_02_regression_importance.png", dpi=180, bbox_inches="tight")
plt.show()
display(Markdown(
    "**Interpretacja.** Permutation importance ocenia całe surowe kolumny przed preprocessingiem, "
    "więc znaczenie kategorii nie jest rozbite na setki dummy variables. Wynik opisuje wpływ predykcyjny, nie przyczynowość."
))
'''),
    md('''## Dodatkowa walidacja czasowa'''),
    code(r'''
latest_year = int(df["Survey Year"].max())
temporal_train = df["Survey Year"] < latest_year
temporal_test = df["Survey Year"] == latest_year
X_time_train, y_time_train = X.loc[temporal_train], y.loc[temporal_train]
X_time_test, y_time_test = X.loc[temporal_test], y.loc[temporal_test]
temporal_model = clone(best_regressor).fit(X_time_train, y_time_train)
temporal_pred = temporal_model.predict(X_time_test)
temporal_result = pd.DataFrame([{
    "train_years": f"{int(df['Survey Year'].min())}-{latest_year - 1}",
    "test_year": latest_year,
    "test_rows": len(y_time_test),
    "MAE_USD": mae_usd(y_time_test, temporal_pred),
    "RMSE_USD": root_mean_squared_error(np.expm1(y_time_test), np.expm1(temporal_pred)),
    "R2_log": r2_score(y_time_test, temporal_pred),
}])
display(temporal_result.round(3))
display(Markdown(
    "**Interpretacja.** Temporal split jest trudniejszym i bardziej realistycznym testem generalizacji na przyszły rok. "
    "`Survey Year` pozostaje cechą, lecz wynagrodzenia są nominalne w USD i nie uwzględniają inflacji ani zmian kursowych."
))
'''),
    md('''
    ## Ograniczenia i zagrożenia trafności regresji

    - badanie jest obserwacyjne i samoopisowe, więc nie pozwala wnioskować przyczynowo;
    - próba respondentów może nie reprezentować całego rynku pracy;
    - wynagrodzenia są nominalne i bez korekty inflacyjnej/PPP;
    - reguła 5 000–500 000 USD ogranicza populację, do której odnoszą się wyniki;
    - losowy split miesza lata, dlatego wynik temporal split raportujemy oddzielnie;
    - strojenie wykorzystuje CV na train; test nie uczestniczy w wyborze hiperparametrów.
    ''')
]


classification_cells = [
    md('''
    # 4. Modelowanie klasyfikacyjne bez sztucznej klasy `Not Asked`

    `Not Asked` nie opisuje planu zawodowego: pytania nie zadawano w 2017 roku. Główna analiza usuwa
    te rekordy, a następnie wykonuje stratified train/test split. Raportujemy balanced accuracy,
    macro F1, precision/recall dla każdej klasy oraz confusion matrix.
    ''') ,
    code(COMMON_SETUP + r'''
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from salary_survey.config import DATA_PROCESSED_PATH, FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from salary_survey.pipelines import build_preprocessor

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
df_all = pd.read_csv(DATA_PROCESSED_PATH)
artifact = pd.crosstab(df_all["Survey Year"], df_all["CareerPlansThisYear"])
display(artifact)
not_asked_count = int((df_all["CareerPlansThisYear"] == "Not Asked").sum())
display(Markdown(
    f"**Decyzja metodologiczna.** Wszystkie {not_asked_count:,} odpowiedzi `Not Asked` pochodzą z 2017 roku. "
    "Usuwamy je, ponieważ brak pytania w kwestionariuszu nie jest zachowaniem respondenta."
))
df = df_all.loc[df_all["CareerPlansThisYear"] != "Not Asked"].copy()
'''),
    md('''## Podział danych i pipeline'''),
    code(r'''
target = "CareerPlansThisYear"
feature_exclusions = [target, "SalaryUSD", "SalaryUSD_Log", "Timestamp", "PostalCode"]
X = df.drop(columns=feature_exclusions, errors="ignore")
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)
display(pd.DataFrame({"train": y_train.value_counts(), "test": y_test.value_counts()}).fillna(0).astype(int))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"}

def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor(X_train)), ("model", model)])

def evaluate_model(name, estimator):
    scores = cross_validate(
        estimator, X_train, y_train, cv=cv, scoring=scoring,
        return_train_score=True, n_jobs=1,
    )
    estimator.fit(X_train, y_train)
    train_pred = estimator.predict(X_train)
    test_pred = estimator.predict(X_test)
    row = {
        "model": name,
        "train_balanced_accuracy": balanced_accuracy_score(y_train, train_pred),
        "CV_balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
        "CV_balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
        "train_macro_F1": f1_score(y_train, train_pred, average="macro"),
        "CV_macro_F1_mean": scores["test_macro_f1"].mean(),
        "CV_macro_F1_std": scores["test_macro_f1"].std(),
        "test_macro_F1": f1_score(y_test, test_pred, average="macro"),
    }
    return estimator, row
'''),
    md('''## Modele domyślne'''),
    code(r'''
models = {
    "Logistic Regression default": make_pipeline(LogisticRegression(
        class_weight="balanced", max_iter=1500, random_state=RANDOM_STATE
    )),
    "Random Forest default": make_pipeline(RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )),
    "LightGBM default": make_pipeline(LGBMClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
    )),
}
classification_rows = []
fitted = {}
for name, estimator in models.items():
    print("Evaluating", name)
    fitted[name], row = evaluate_model(name, estimator)
    classification_rows.append(row)
display(pd.DataFrame(classification_rows).round(4))
'''),
    md('''## Strojenie LightGBM i porównanie default vs tuned'''),
    code(r'''
lgb_pipeline = make_pipeline(LGBMClassifier(
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
))
lgb_params = {
    "model__n_estimators": [100, 180, 260],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__num_leaves": [15, 31, 63],
    "model__max_depth": [-1, 6, 10],
    "model__min_child_samples": [10, 20, 40],
}
lgb_search = RandomizedSearchCV(
    lgb_pipeline, param_distributions=lgb_params, n_iter=8, scoring="f1_macro",
    cv=3, random_state=RANDOM_STATE, n_jobs=1, return_train_score=True, verbose=1,
)
lgb_search.fit(X_train, y_train)
print("Best parameters:", lgb_search.best_params_)
best_classifier, tuned_row = evaluate_model("LightGBM tuned", lgb_search.best_estimator_)
classification_rows.append(tuned_row)
fitted["LightGBM tuned"] = best_classifier

classification_results = (pd.DataFrame(classification_rows).set_index("model")
                          .sort_values("CV_macro_F1_mean", ascending=False))
best_model_name = classification_results.index[0]
best_classifier = fitted[best_model_name]
display(classification_results.round(4))
print("Selected by CV Macro F1:", best_model_name)
classification_results.to_csv(PROJECT_ROOT / "reports" / "classification_model_comparison.csv")
joblib.dump(best_classifier, MODELS_DIR / "classification_best_pipeline.joblib")
'''),
    md('''## Porównanie modeli klasyfikacyjnych'''),
    code(r'''
plot_data = (classification_results.reset_index()
             .melt(id_vars="model", value_vars=["train_macro_F1", "CV_macro_F1_mean", "test_macro_F1"],
                   var_name="split", value_name="macro_F1"))
plt.figure(figsize=(12, 6))
sns.barplot(data=plot_data, x="model", y="macro_F1", hue="split")
plt.title("Porównanie modeli klasyfikacyjnych — Macro F1")
plt.ylim(0, 1)
plt.xlabel("")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_03_classification_comparison.png", dpi=180, bbox_inches="tight")
plt.show()
display(Markdown(
    "**Interpretacja.** Macro F1 nadaje każdej prawdziwej kategorii planów równą wagę, a balanced accuracy "
    "uśrednia recall klas. Brak sztucznej klasy `Not Asked` usuwa wcześniejszy automatyczny wynik F1=1.00."
))
'''),
    md('''## Per-class precision/recall oraz confusion matrix'''),
    code(r'''
test_pred = best_classifier.predict(X_test)
report = pd.DataFrame(classification_report(y_test, test_pred, output_dict=True, zero_division=0)).T
display(report.round(3))
report.to_csv(PROJECT_ROOT / "reports" / "classification_per_class_metrics.csv")

labels = sorted(y.unique())
cm = confusion_matrix(y_test, test_pred, labels=labels, normalize="true")
fig, ax = plt.subplots(figsize=(11, 8))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(
    ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=35, colorbar=False
)
ax.set_title("Znormalizowana confusion matrix — klasyfikacja bez Not Asked")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_04_classification_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()
display(Markdown(
    "**Interpretacja.** Wiersze sumują się do 1, dlatego przekątna pokazuje recall każdej klasy. "
    "Macierz ujawnia, które rzeczywiste plany są mylone, czego sama accuracy nie pokazuje."
))
'''),
    md('''## Interpretacja modelu klasyfikacyjnego'''),
    code(r'''
sample_idx = X_test.sample(min(1200, len(X_test)), random_state=RANDOM_STATE).index
importance = permutation_importance(
    best_classifier, X_test.loc[sample_idx], y_test.loc[sample_idx],
    scoring="f1_macro", n_repeats=4, random_state=RANDOM_STATE, n_jobs=1,
)
importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
}).sort_values("importance_mean", ascending=False).head(15)
display(importance_df)
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="importance_mean", y="feature", color="#59A14F")
plt.title("Permutation importance — klasyfikacja planów zawodowych")
plt.xlabel("Spadek Macro F1 po permutacji")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_05_classification_importance.png", dpi=180, bbox_inches="tight")
plt.show()
display(Markdown(
    "**Interpretacja.** Wykres pokazuje cechy przydatne predykcyjnie po usunięciu artefaktu 2017. "
    "Nie należy interpretować ich jako przyczyn decyzji zawodowych."
))
'''),
    md('''
    ## Ograniczenia klasyfikacji

    - `Not Asked` usunięto jako artefakt wersji ankiety; wyniki dotyczą lat, w których pytanie zadawano;
    - klasy pozostają niezbalansowane, dlatego accuracy nie jest metryką główną;
    - odpowiedzi deklaratywne nie muszą odpowiadać późniejszym zachowaniom;
    - preprocessing i tuning korzystają wyłącznie z train/CV; test pozostaje końcową oceną;
    - permutation importance opisuje predykcję, nie zależność przyczynową.
    ''')
]


write_notebook("03_modeling_regression.ipynb", regression_cells)
write_notebook("04_modeling_classification.ipynb", classification_cells)
print("Generated regression and classification notebooks")


clustering_cells = [
    md('''
    # 5. Klasteryzacja i profilowanie segmentów

    Klasteryzacja nie używa zmiennych docelowych (`SalaryUSD`, `CareerPlansThisYear`).
    Liczbę klastrów K-Means wybieramy dla `k=2...10` przez inertia i silhouette.
    TruncatedSVD jest oceniane przez explained variance; 2 komponenty służą wyłącznie do wizualizacji,
    natomiast algorytmy korzystają z wielowymiarowej reprezentacji.
    ''') ,
    code(COMMON_SETUP + r'''
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from salary_survey.config import DATA_PROCESSED_PATH, FIGURES_DIR, RANDOM_STATE
from salary_survey.pipelines import build_preprocessor

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
df = pd.read_csv(DATA_PROCESSED_PATH)
exclude = ["SalaryUSD", "SalaryUSD_Log", "CareerPlansThisYear", "Timestamp", "PostalCode"]
X_raw = df.drop(columns=exclude, errors="ignore")
preprocessor = build_preprocessor(X_raw)
X_encoded = preprocessor.fit_transform(X_raw)
print("Encoded clustering matrix:", X_encoded.shape)
'''),
    md('''## Explained variance i wybór liczby komponentów'''),
    code(r'''
max_components = min(50, X_encoded.shape[1] - 1)
svd_full = TruncatedSVD(n_components=max_components, random_state=RANDOM_STATE)
X_svd_full = svd_full.fit_transform(X_encoded)
cumulative_variance = np.cumsum(svd_full.explained_variance_ratio_)
variance_table = pd.DataFrame({
    "component": np.arange(1, max_components + 1),
    "explained_variance": svd_full.explained_variance_ratio_,
    "cumulative_variance": cumulative_variance,
})
display(variance_table.head(15).round(4))

plt.figure(figsize=(10, 5))
plt.plot(variance_table["component"], variance_table["cumulative_variance"], marker="o", markersize=3)
plt.axhline(0.80, color="red", linestyle="--", label="80% wariancji")
plt.title("TruncatedSVD — skumulowana explained variance")
plt.xlabel("Liczba komponentów")
plt.ylabel("Skumulowany udział wyjaśnionej wariancji")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cluster_01_svd_explained_variance.png", dpi=180, bbox_inches="tight")
plt.show()
display(Markdown(
    "**Interpretacja.** Krzywa pokazuje koszt redukcji wymiaru. Dwa komponenty nie zachowują całej struktury, "
    "dlatego do klasteryzacji wykorzystujemy więcej wymiarów; komponenty 1–2 pozostają tylko mapą poglądową."
))

n_cluster_components = min(20, max_components)
X_cluster = StandardScaler().fit_transform(X_svd_full[:, :n_cluster_components])
print(f"Clustering uses {n_cluster_components} SVD components; 2D is visualization only.")
'''),
    md('''## K-Means: elbow curve i silhouette dla k=2...10'''),
    code(r'''
k_rows = []
k_models = {}
for k in range(2, 11):
    model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_cluster)
    score = silhouette_score(
        X_cluster, labels, sample_size=min(4000, len(X_cluster)), random_state=RANDOM_STATE
    )
    min_cluster_pct = 100 * np.bincount(labels).min() / len(labels)
    k_rows.append({"k": k, "inertia": model.inertia_, "silhouette": score,
                   "min_cluster_pct": min_cluster_pct})
    k_models[k] = (model, labels)
k_scores = pd.DataFrame(k_rows)
display(k_scores.round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.lineplot(data=k_scores, x="k", y="inertia", marker="o", ax=axes[0])
sns.lineplot(data=k_scores, x="k", y="silhouette", marker="o", ax=axes[1])
axes[0].set_title("Elbow curve")
axes[1].set_title("Silhouette score")
axes[0].set_xticks(range(2, 11))
axes[1].set_xticks(range(2, 11))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cluster_02_k_selection.png", dpi=180, bbox_inches="tight")
plt.show()

qualified = k_scores.loc[k_scores["min_cluster_pct"] >= 2.0]
best_k = int(qualified.loc[qualified["silhouette"].idxmax(), "k"])
kmeans, kmeans_labels = k_models[best_k]
df["Cluster_KMeans"] = kmeans_labels
display(Markdown(
    f"**Interpretacja.** Spośród rozwiązań bez klastrów mniejszych niż 2% próby najwyższy silhouette uzyskano dla k={best_k}. "
    "Wszystkie wartości silhouette są niskie, więc struktura segmentów jest słaba; ograniczenie minimalnego rozmiaru "
    "zapobiega wyborowi pozornie lepszego rozwiązania z kilkoma klastrami złożonymi z pojedynczych obserwacji."
))
'''),
    md('''## Wizualizacja 2D wybranego K-Means'''),
    code(r'''
plot_frame = pd.DataFrame({
    "SVD_1": X_svd_full[:, 0], "SVD_2": X_svd_full[:, 1], "cluster": kmeans_labels.astype(str)
})
plt.figure(figsize=(10, 7))
sns.scatterplot(data=plot_frame.sample(min(6000, len(plot_frame)), random_state=RANDOM_STATE),
                x="SVD_1", y="SVD_2", hue="cluster", alpha=0.45, s=20, palette="tab10")
plt.title(f"K-Means (k={best_k}) — rzut na pierwsze 2 komponenty SVD")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cluster_03_kmeans_2d.png", dpi=180, bbox_inches="tight")
plt.show()
display(Markdown(
    "**Interpretacja.** Rzut 2D pomaga zobaczyć nakładanie segmentów, ale model był dopasowany w przestrzeni "
    f"{n_cluster_components}-wymiarowej. Pozorne nakładanie na wykresie nie oznacza automatycznie braku separacji w pozostałych wymiarach."
))
'''),
    md('''## Profile klastrów'''),
    code(r'''
def mode_or_unknown(series):
    mode = series.dropna().mode()
    return mode.iloc[0] if len(mode) else "Unknown"

profiles = (df.groupby("Cluster_KMeans")
    .agg(
        size=("SalaryUSD", "size"),
        median_salary=("SalaryUSD", "median"),
        median_job_years=("YearsWithThisTypeOfJob", "median"),
        median_database_years=("YearsWithThisDatabase", "median"),
        dominant_country=("Country", mode_or_unknown),
        dominant_job=("JobTitle", mode_or_unknown),
        dominant_employment=("EmploymentStatus", mode_or_unknown),
    )
)
profiles["share_pct"] = 100 * profiles["size"] / profiles["size"].sum()
display(profiles.round(2))
profiles.to_csv(PROJECT_ROOT / "reports" / "cluster_profiles.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=profiles.reset_index(), x="Cluster_KMeans", y="median_salary", ax=axes[0])
sns.barplot(data=profiles.reset_index(), x="Cluster_KMeans", y="median_job_years", ax=axes[1])
axes[0].set_title("Mediana SalaryUSD w klastrach")
axes[1].set_title("Mediana stażu zawodowego w klastrach")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cluster_04_profiles.png", dpi=180, bbox_inches="tight")
plt.show()
display(Markdown(
    "**Interpretacja.** Profilowanie przez kraj, stanowisko, wynagrodzenie i staż nadaje segmentom treść. "
    "SalaryUSD nie uczestniczyło w klasteryzacji, więc różnice wynagrodzeń są opisem ex post, a nie wymuszoną separacją."
))
'''),
    md('''## DBSCAN: sprawdzenie parametrów i interpretacja wyniku negatywnego'''),
    code(r'''
dbscan_rows = []
dbscan_models = {}
for eps in [0.5, 0.8, 1.1, 1.5, 2.0]:
    for min_samples in [10, 20, 40]:
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_cluster)
        non_noise = labels != -1
        n_clusters = len(set(labels[non_noise]))
        noise_pct = 100 * (~non_noise).mean()
        score = np.nan
        if n_clusters > 1 and non_noise.sum() > n_clusters:
            score = silhouette_score(
                X_cluster[non_noise], labels[non_noise],
                sample_size=min(3000, non_noise.sum()), random_state=RANDOM_STATE,
            )
        dbscan_rows.append({"eps": eps, "min_samples": min_samples,
                            "clusters": n_clusters, "noise_pct": noise_pct, "silhouette": score})
        dbscan_models[(eps, min_samples)] = labels
dbscan_results = pd.DataFrame(dbscan_rows)
display(dbscan_results.round(4))

valid = dbscan_results.dropna(subset=["silhouette"])
if valid.empty:
    dbscan_conclusion = (
        "Żaden sprawdzony wariant DBSCAN nie utworzył co najmniej dwóch stabilnych klastrów. "
        "To wynik negatywny: dane nie wykazują wyraźnej struktury gęstościowej przy badanym skalowaniu."
    )
else:
    best = valid.loc[valid["silhouette"].idxmax()]
    dbscan_conclusion = (
        f"Najlepszy badany DBSCAN: eps={best.eps}, min_samples={int(best.min_samples)}, "
        f"klastry={int(best.clusters)}, silhouette={best.silhouette:.3f}, szum={best.noise_pct:.1f}%."
    )
display(Markdown(f"**Interpretacja.** {dbscan_conclusion} DBSCAN nie jest przedstawiany jako udana segmentacja bez dowodów metrycznych."))
'''),
    md('''
    ## Ograniczenia klasteryzacji

    - wyniki zależą od kodowania, skalowania oraz liczby komponentów SVD;
    - silhouette mierzy separację geometryczną, nie użyteczność biznesową;
    - K-Means preferuje klastry zbliżone do kulistych;
    - profile są opisowe, a nie przyczynowe;
    - negatywny wynik DBSCAN jest ważnym rezultatem i nie stanowi drugiej udanej segmentacji.
    ''')
]


write_notebook("05_modeling_clustering.ipynb", clustering_cells)
print("Generated clustering notebook")
