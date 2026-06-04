"""Эксперимент по сравнению SVM, Fuzzy k-NN и AutoML-кандидатов.

Скрипт загружает данные диспансеризации, готовит признаки, подбирает
гиперпараметры моделей через GridSearchCV и сохраняет итоговые метрики.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Фиксируем random_state, чтобы разбиение данных и обучение можно было повторить.
RANDOM_STATE = 42
DATA_PATH = Path("data/dispensarization_data_2026.csv")
OUTPUT_DIR = Path("automl_report_assets")
RESULTS_PATH = OUTPUT_DIR / "results" / "experiment_results.json"
ROOT_RESULTS_PATH = Path("experiment_results.json")

TARGET = "ССЗ_риск_высокий"

# Эти колонки нельзя использовать как признаки:
# TARGET — это ответ, а две остальные колонки дают слишком прямую подсказку модели.
LEAKAGE_COLUMNS = {TARGET, "Статус_глюкозы", "Доклинический_риск"}


class FuzzyKNNClassifier(ClassifierMixin, BaseEstimator):
    """Fuzzy k-NN classifier with distance-weighted class memberships."""

    def __init__(self, n_neighbors: int = 15, m: float = 2.0):
        self.n_neighbors = n_neighbors
        self.m = m

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuzzyKNNClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if self.m <= 1:
            raise ValueError("m must be greater than 1")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")

        # Fuzzy k-NN не строит сложную модель: он запоминает train-объекты
        # и потом сравнивает с ними новых пациентов.
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self._class_to_index_ = {klass: i for i, klass in enumerate(self.classes_)}
        self._effective_neighbors_ = min(self.n_neighbors, len(X))
        self._nn_ = NearestNeighbors(n_neighbors=self._effective_neighbors_)
        self._nn_.fit(X)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        # Для каждого нового пациента ищем ближайших пациентов из train.
        distances, indices = self._nn_.kneighbors(X, return_distance=True)
        probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)

        exponent = 2.0 / (self.m - 1.0)
        for row_idx, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            zero_mask = row_distances == 0
            if np.any(zero_mask):
                # Если нашли полностью совпадающего пациента, используем его класс напрямую.
                neighbor_labels = self.y_[row_indices[zero_mask]]
                for label in neighbor_labels:
                    probabilities[row_idx, self._class_to_index_[label]] += 1.0
                probabilities[row_idx] /= zero_mask.sum()
                continue

            # Чем ближе сосед, тем больше его вес в вероятности класса.
            weights = 1.0 / np.power(row_distances, exponent)
            for label, weight in zip(self.y_[row_indices], weights):
                probabilities[row_idx, self._class_to_index_[label]] += weight
            probabilities[row_idx] /= weights.sum()

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def to_jsonable(value: Any) -> Any:
    """Преобразует numpy-типы в обычные Python-типы для сохранения в JSON."""

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Простая конвертация таблицы pandas в markdown без внешних зависимостей."""

    headers = [str(column) for column in df.columns]
    rows = []
    for row in df.itertuples(index=False):
        rows.append([str(value) for value in row])

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_pipeline(classifier: Any) -> Pipeline:
    """Создаёт общий пайплайн: заполнение пропусков, масштабирование, модель."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def positive_class_score(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Возвращает скор класса 1, который нужен для ROC-AUC."""

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        return probabilities[:, 1]
    scores = estimator.decision_function(X)
    return np.asarray(scores, dtype=float)


def evaluate_model(
    name: str,
    estimator: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> tuple[dict[str, Any], dict[str, Any], GridSearchCV]:
    """Подбирает параметры модели и считает метрики на test-выборке."""

    # GridSearchCV перебирает заданную сетку параметров и выбирает лучший вариант
    # по ROC-AUC на кросс-валидации.
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    # Обучаем модель на train и измеряем время подбора.
    fit_started = time.perf_counter()
    search.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_started

    # Финальная проверка идёт на test, который не участвовал в обучении.
    predict_started = time.perf_counter()
    y_pred = search.predict(X_test)
    y_score = positive_class_score(search.best_estimator_, X_test)
    predict_time = time.perf_counter() - predict_started

    # Метрики считаем сразу в одном месте, чтобы все модели сравнивались одинаково.
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, y_score), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "fit_time_sec": round(fit_time, 2),
        "predict_time_sec": round(predict_time, 4),
        "cv_roc_auc": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # Сохраняем не только метрики, но и сами предсказания для графиков.
    predictions = {
        "model": name,
        "y_true": y_test.astype(int).tolist(),
        "y_pred": np.asarray(y_pred, dtype=int).tolist(),
        "y_score": np.asarray(y_score, dtype=float).round(8).tolist(),
    }
    return metrics, predictions, search


def make_eda_summary(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """Собирает краткое описание датасета для отчёта."""

    target_counts = df[TARGET].value_counts().sort_index()
    missing_pct = (
        df[features]
        .isna()
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )
    missing_top = missing_pct[missing_pct > 0].head(8).to_dict()

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_features": int(len(features)),
        "age_min": int(df["Возраст"].min()),
        "age_max": int(df["Возраст"].max()),
        "target": TARGET,
        "class_0": int(target_counts.get(0, 0)),
        "class_1": int(target_counts.get(1, 0)),
        "class_1_pct": round(float(target_counts.get(1, 0) / len(df) * 100), 1),
        "missing_top_pct": missing_top,
        "bmi_mean": round(float(df["ИМТ"].mean()), 2),
        "glucose_mean": round(float(df["Глюкоза_натощак_ммоль_л"].mean()), 2),
        "sbp_mean": round(float(df["САД_мм_рт_ст"].mean()), 2),
        "features": features,
    }


def main() -> None:
    # Создаём папки, куда будут сохранены JSON, таблицы и результаты.
    OUTPUT_DIR.joinpath("results").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)

    # Загружаем датасет и отделяем признаки X от целевой переменной y.
    df = pd.read_csv(DATA_PATH)
    features = [column for column in df.columns if column not in LEAKAGE_COLUMNS]
    X = df[features]
    y = df[TARGET]

    # Делим данные на train/test 80/20.
    # stratify=y сохраняет долю пациентов высокого риска в обеих частях.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # На train используем 5-fold стратифицированную кросс-валидацию.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Список моделей для эксперимента.
    # manual_baseline — ручные базовые модели.
    # automl_candidate — кандидаты, среди которых автоматически выбираем лучшую модель.
    model_specs = [
        (
            "SVM (GridSearch)",
            build_pipeline(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            {
                "clf__C": [0.1, 1, 10, 100],
                "clf__gamma": ["scale", 0.1, 0.01, 0.001],
            },
            "manual_baseline",
        ),
        (
            "Fuzzy k-NN (GridSearch)",
            build_pipeline(FuzzyKNNClassifier()),
            {
                "clf__n_neighbors": [5, 9, 15, 25, 35],
                "clf__m": [1.5, 2.0, 2.5],
            },
            "manual_baseline",
        ),
        (
            "AutoML: LogisticRegression",
            build_pipeline(
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                )
            ),
            {"clf__C": [0.1, 1, 10]},
            "automl_candidate",
        ),
        (
            "AutoML: RandomForest",
            build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 3],
            },
            "automl_candidate",
        ),
        (
            "AutoML: GradientBoosting",
            build_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
            {
                "clf__n_estimators": [50, 100, 150],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
            "automl_candidate",
        ),
        (
            "AutoML: KNN",
            build_pipeline(KNeighborsClassifier()),
            {"clf__n_neighbors": [5, 9, 15, 25, 35]},
            "automl_candidate",
        ),
        (
            "AutoML: SVM",
            build_pipeline(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.1, 0.01],
            },
            "automl_candidate",
        ),
    ]

    model_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    automl_started = time.perf_counter()

    # Последовательно обучаем каждую модель, подбираем параметры и считаем метрики.
    for name, estimator, param_grid, category in model_specs:
        metrics, prediction, _ = evaluate_model(
            name,
            estimator,
            param_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            cv,
        )
        metrics["category"] = category
        model_metrics.append(metrics)
        predictions.append(prediction)
        print(f"{name}: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")

    # Суммарное время считаем отдельно для AutoML-кандидатов.
    automl_full_time = round(
        sum(item["fit_time_sec"] for item in model_metrics if item["category"] == "automl_candidate"),
        2
    )
    wall_time_all = round(time.perf_counter() - automl_started, 2)

    # Главный AutoML-выбор: берём кандидата с максимальным ROC-AUC.
    best_automl = max(
        (item for item in model_metrics if item["category"] == "automl_candidate"),
        key=lambda item: item["roc_auc"],
    )

    # Сравниваем лучший AutoML-кандидат с ручным SVM для проверки гипотезы.
    svm = next(item for item in model_metrics if item["model"] == "SVM (GridSearch)")
    auc_gain_pct_points = round((best_automl["roc_auc"] - svm["roc_auc"]) * 100, 2)

    # Собираем всё в один словарь: данные, EDA, метрики, предсказания, гипотезу.
    results = {
        "dataset": str(DATA_PATH),
        "task": "Бинарная классификация высокого сердечно-сосудистого риска",
        "target": TARGET,
        "split": "train/test 80/20, stratified, random_state=42",
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring=roc_auc",
        "eda": make_eda_summary(df, features),
        "models": model_metrics,
        "predictions": predictions,
        "automl_full_fit_time_sec": automl_full_time,
        "wall_time_all_sec": wall_time_all,
        "hypothesis": {
            "text": "Лучший AutoML-кандидат даст прирост ROC-AUC к настроенному SVM не более 5 процентных пунктов.",
            "best_automl_model": best_automl["model"],
            "svm_roc_auc": svm["roc_auc"],
            "best_automl_roc_auc": best_automl["roc_auc"],
            "auc_gain_pct_points": auc_gain_pct_points,
            "confirmed": auc_gain_pct_points <= 5,
        },
    }

    # Полный JSON нужен для воспроизведения результатов и построения графиков.
    RESULTS_PATH.write_text(json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    ROOT_RESULTS_PATH.write_text(json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")

    # Отдельно сохраняем удобную таблицу метрик в CSV и markdown.
    model_table = pd.DataFrame(model_metrics)
    table_columns = [
        "model",
        "roc_auc",
        "f1",
        "balanced_accuracy",
        "accuracy",
        "precision",
        "recall",
        "fit_time_sec",
        "predict_time_sec",
        "cv_roc_auc",
        "best_params",
        "category",
    ]
    model_table = model_table[table_columns]
    model_table.to_csv(OUTPUT_DIR / "tables" / "model_comparison.csv", index=False)
    (OUTPUT_DIR / "tables" / "model_comparison.md").write_text(
        dataframe_to_markdown(model_table),
        encoding="utf-8",
    )

    print(f"\nSaved results: {RESULTS_PATH}")
    print(f"Saved table: {OUTPUT_DIR / 'tables' / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
