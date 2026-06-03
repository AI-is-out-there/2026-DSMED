import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from flaml import AutoML

import h2o
from h2o.automl import H2OAutoML


# первичный анализ
df = pd.read_csv("2022.csv")
print("Первые 5 строк")
print(df.head())


print("Размер датасета:")
print(df.shape)


print("Информация о данных:")
print(df.info())


print("Статистика:")
print(df.describe())

# ==========================================
# ПРЕОБРАЗОВАНИЕ КЛАССОВ
# ==========================================

df["Стадия"] = df["Стадия"].map({
    "Бодрствование": 0,
    "Лёгкий сон": 1,
    "Глубокий сон": 2,
    "Пробуждение": 3
})

print("\nРаспределение классов:")
print(df["Стадия"].value_counts())


# ПЕРВИЧНАЯ ОБРАБОТКА


if "Час" in df.columns:
    df["night"] = (
        (df["Час"] >= 22) |
        (df["Час"] <= 6)
    ).astype(int)

df["hr_diff"] = df["heartRate"].diff().fillna(0)

drop_cols = ["full_time", "date", "time"]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

df.dropna(inplace=True)


# TRAIN / TEST


train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Стадия"]
)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

X_train = train.drop("Стадия", axis=1)
y_train = train["Стадия"]

X_test = test.drop("Стадия", axis=1)
y_test = test["Стадия"]


# BASELINE
y_test_binary = (y_test != 0).astype(int)

baseline_pred = (
    X_test["heartRate"] <= 80
).astype(int)

baseline_acc = accuracy_score(
    y_test_binary,
    baseline_pred
)


print("BASELINE")
print(f"Accuracy = {baseline_acc:.4f}")


# FLAML AutoML
print("FLAML AutoML")

automl = AutoML()

automl.fit(
    X_train,
    y_train,
    task="classification",
    time_budget=180
)

flaml_pred = automl.predict(X_test)

flaml_acc = accuracy_score(
    y_test,
    flaml_pred
)

print(f"Accuracy = {flaml_acc:.4f}")
print("Лучшая модель:", automl.best_estimator)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        flaml_pred
    )
)


# H2O AutoML



print("H2O AutoML")

h2o.init()

train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)

train_h2o["Стадия"] = train_h2o["Стадия"].asfactor()
test_h2o["Стадия"] = test_h2o["Стадия"].asfactor()

aml = H2OAutoML(
    max_runtime_secs=180,
    seed=42
)

aml.train(
    y="Стадия",
    training_frame=train_h2o
)

print("Лучшая модель:", aml.leader.model_id)

pred = aml.leader.predict(test_h2o)

pred_df = pred.as_data_frame()

print("\nРазмеры:")
print("test:", len(test))
print("pred:", len(pred_df))

h2o_pred = pred_df["predict"].astype(int).to_numpy()

y_test_np = y_test.astype(int).to_numpy()
n = min(len(y_test_np), len(h2o_pred))

h2o_acc = accuracy_score(
    y_test_np[:n],
    h2o_pred[:n]
)

print(f"Accuracy = {h2o_acc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test_np[:n],
        h2o_pred[:n]
    )
)


# СРАВНЕНИЕ

print("ИТОГОВОЕ СРАВНЕНИЕ")

print(f"Baseline : {baseline_acc:.4f}")
print(f"FLAML    : {flaml_acc:.4f}")
print(f"H2O      : {h2o_acc:.4f}")

print("\nМатрица ошибок FLAML:")
print(
    confusion_matrix(
        y_test,
        flaml_pred
    )
)

h2o.cluster().shutdown(prompt=False)