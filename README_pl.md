# Zastosowanie metod uczenia maszynowego - Projekt Zaliczeniowy
*Przeczytaj w innym języku: [🇬🇧 English](README.md)*

## 📌 Opis Projektu
Projekt realizowany w ramach przedmiotu **Zastosowanie metod uczenia maszynowego**. Głównym celem jest przeprowadzenie kompletnego procesu analizy danych (end-to-end machine learning pipeline) na zbiorze "Data Professional Salary Survey". 

Projekt rozwiązuje dwa główne problemy biznesowe:
1. **Regresja:** Przewidywanie rocznego wynagrodzenia specjalistów IT (`SalaryUSD`).
2. **Klasyfikacja:** Przewidywanie planów zawodowych na bieżący rok (`CareerPlansThisYear`).

## 🎯 Wymagania i Cele
Zgodnie z wytycznymi, projekt musi spełniać następujące kryteria i zawierać minimum **8 różnych modeli uczenia maszynowego** (4 dla regresji, 4 dla klasyfikacji). 

Składowe oceny końcowej:
* Eksploracyjna analiza danych (EDA)
* Wizualizacja danych
* Przygotowanie danych (Czyszczenie, obsługa braków)
* Inżynieria cech (Feature Engineering)
* Budowa modeli (Min. 8 modeli algorytmicznych)
* Strojenie hiperparametrów (GridSearchCV / Optuna)
* Porównanie wyników i walidacja modeli
* Interpretacja modeli (wykorzystanie biblioteki SHAP)
* Raport końcowy (PDF) i wnioski biznesowe

## 📊 Zbiór Danych
Dane pochodzą z ankiety **Brent Ozar Unlimited Data Professional Salary Survey**. 
Ze względu na charakter odpowiedzi ankietowych (wpisywane ręcznie przez użytkowników), zbiór jest mocno zanieczyszczony i wymaga zaawansowanych technik czyszczenia (m.in. detekcja wartości odstających, ujednolicenie zmiennych kategorycznych).

## 🛠 Technologie i Narzędzia
* **Język:** Python 3.x
* **Eksploracja i transformacja:** Pandas, NumPy
* **Wizualizacja:** Matplotlib, Seaborn
* **Modele ML:** Scikit-Learn, XGBoost, LightGBM
* **Wyjaśnialność modeli (XAI):** SHAP
* **Środowisko:** Jupyter Notebook

## 🚀 Plan Działania (Roadmap / Co trzeba zrobić)

### Faza 1: Przygotowanie i EDA
- [x] Inicjalizacja repozytorium GitHub i środowiska pracy.
- [ ] Wczytanie danych i wstępne profilowanie.
- [ ] Czyszczenie zmiennej docelowej `SalaryUSD` (usunięcie ekstremalnych wartości odstających).
- [ ] Czyszczenie zmiennej docelowej `CareerPlansThisYear` (grupowanie klas).
- [ ] Obsługa braków danych (imputacja) oraz kodowanie zmiennych kategorycznych (One-Hot / Label Encoding).
- [ ] Zaawansowana wizualizacja (Macierz korelacji, rozkłady wynagrodzeń wg technologii i krajów).

### Faza 2: Regresja (Przewidywanie `SalaryUSD`)
- [ ] Model 1: Linear Regression (Baseline).
- [ ] Model 2: Decision Tree Regressor.
- [ ] Model 3: Random Forest Regressor.
- [ ] Model 4: XGBoost Regressor.
- [ ] Strojenie hiperparametrów (Hyperparameter tuning) dla najlepszych modeli.
- [ ] Ocena modeli (RMSE, MAE, R2).
- [ ] Interpretacja wyników (Feature Importance / SHAP).

### Faza 3: Klasyfikacja (Przewidywanie `CareerPlansThisYear`)
- [ ] Rozwiązanie problemu niezbalansowanych klas (np. SMOTE, class weights).
- [ ] Model 1: Logistic Regression (Baseline).
- [ ] Model 2: K-Nearest Neighbors (KNN).
- [ ] Model 3: Random Forest Classifier.
- [ ] Model 4: LightGBM / XGBoost Classifier.
- [ ] Strojenie hiperparametrów.
- [ ] Ocena modeli (Precision, Recall, F1-score, Confusion Matrix).
- [ ] Interpretacja wyników (Feature Importance / SHAP).

### Faza 4: Finalizacja
- [ ] Opracowanie wniosków końcowych.
- [ ] Wygenerowanie profesjonalnego raportu PDF (10-20 stron).
- [ ] Przesłanie plików na MS Teams zgodnie z wytycznymi prowadzącego.

---
*Autor: [Twoje Imię i Nazwisko]*
