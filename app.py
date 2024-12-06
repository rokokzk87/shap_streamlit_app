import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import shap
import streamlit as st
import matplotlib.pyplot as plt

# Кэшируем загрузку и обработку данных
@st.cache_data
def load_and_process_data():
    # Загрузка данных
    df = pd.read_csv('training.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Разделение данных на признаки и целевую переменную
    X = df.drop(columns=['date', 'target'])
    y = df['target']

    # Обработка пропущенных значений
    X = X.fillna(0)

    return df, X, y

# Кэшируем обучение модели
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Кэшируем создание объяснителя SHAP
@st.cache_resource
def create_shap_explainer(_model):
    explainer = shap.TreeExplainer(_model)
    return explainer


st.title("Визуализация влияния признаков с помощью SHAP")

st.write("""
    Это приложение демонстрирует, как каждый признак влияет на предсказания модели.
    Вы можете выбрать общий обзор или проанализировать конкретную дату.
""")

# Загрузка и обработка данных
df, X, y = load_and_process_data()

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = train_model(X_train, y_train)

# Создание объяснителя SHAP
explainer = create_shap_explainer(model)

# Вычисление SHAP-значений для тестовой выборки
shap_values = explainer.shap_values(X_test)

# Инициализация Session State для метрик
if 'show_metrics' not in st.session_state:
    st.session_state.show_metrics = False

# Обработка нажатия кнопки метрик в sidebar
if st.sidebar.button("Показать метрики точности модели"):
    st.session_state.show_metrics = True

# Отображение метрик, если кнопка была нажата
if st.session_state.show_metrics:
    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test)

    # Вычисление метрик
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Отображение метрик
    st.sidebar.write("**Метрики модели на тестовой выборке:**")
    st.sidebar.write(f"**R² (коэффициент детерминации):** {r2:.4f}")
    st.sidebar.write(f"**MAE (средняя абсолютная ошибка):** {mae:.4f}")
    st.sidebar.write(f"**MSE (средняя квадратичная ошибка):** {mse:.4f}")
    st.sidebar.write(f"**RMSE (корень из MSE):** {rmse:.4f}")

# Выбор режима визуализации
option = st.selectbox('Выберите тип визуализации', ('Summary Plot', 'Waterfall Plot'))

if option == 'Summary Plot':
    if st.button("Показать SHAP Summary Plot"):
        st.subheader("SHAP Summary Plot")
        st.write("Этот график показывает важность признаков и их влияние на предсказания модели.")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        st.pyplot(plt)

elif option == 'Waterfall Plot':

    selected_date = st.date_input("Выберите дату", df['date'].min())

    # Фильтрация данных по выбранной дате
    df_filtered = df[df['date'] == pd.to_datetime(selected_date)]

    if df_filtered.empty:
        st.write("Нет данных за выбранную дату.")
    else:
        # Если несколько записей за дату, позволим выбрать одну
        if len(df_filtered) > 1:
            selected_record = st.selectbox(
                "Выберите запись",
                df_filtered.index,
                format_func=lambda x: f"Запись {x}"
            )
            selected_row = df_filtered.loc[selected_record]
        else:
            selected_row = df_filtered.iloc[0]

        # Подготовка данных для SHAP
        X_selected = selected_row.drop(labels=['date', 'target']).to_frame().T
        X_selected = X_selected.fillna(0)

        # Вычисление SHAP-значений для выбранного экземпляра
        shap_values_selected = explainer.shap_values(X_selected)

        # Предсказание модели для выбранного экземпляра
        prediction = model.predict(X_selected)

        # Получение реального значения целевой переменной
        actual_value = selected_row['target']

        st.write(f"**Предсказание модели для выбранной даты:** {prediction[0]:.2f}")
        st.write(f"**Реальное значение:** {actual_value}")

        if st.button("Показать SHAP Waterfall Plot"):
            st.write("**SHAP Waterfall Plot:**")
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_selected[0],
                    base_values=explainer.expected_value,
                    data=X_selected.iloc[0],
                    feature_names=X_selected.columns
                ),
                show=False
            )
            plt.tight_layout()
            st.pyplot(plt)

