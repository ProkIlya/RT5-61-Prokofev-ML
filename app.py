import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Прогнозирование качества вина",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #722F37; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .prediction-good { color: #27ae60; font-weight: bold; font-size: 1.5rem; }
    .prediction-bad { color: #e74c3c; font-weight: bold; font-size: 1.5rem; }
    .stSlider > div > div > div { background-color: #722F37 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🍷 Прогнозирование качества вина</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Бинарная классификация: хорошее вино (quality ≥ 6) против плохого</div>', unsafe_allow_html=True)

# Загрузка и подготовка данных
@st.cache_data
def load_and_preprocess():
    url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    df_red = pd.read_csv(url_red, sep=';')
    df_white = pd.read_csv(url_white, sep=';')
    df_red['wine_type'] = 'red'
    df_white['wine_type'] = 'white'
    df = pd.concat([df_red, df_white], axis=0, ignore_index=True)
    
    # Целевая переменная (как в ноутбуке)
    df['quality_binary'] = (df['quality'] >= 6).astype(int)
    
    # Кодирование типа вина
    le = LabelEncoder()
    df['wine_type_encoded'] = le.fit_transform(df['wine_type'])
    
    # Инжиниринг признаков
    df['free_to_total_sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)
    df['acid_balance'] = df['fixed acidity'] / (df['volatile acidity'] + 1e-6)
    df['sugar_alcohol_ratio'] = df['residual sugar'] / (df['alcohol'] + 1e-6)
    df['ph_category'] = pd.cut(df['pH'], bins=[0, 3.0, 3.5, 4.0], labels=['low', 'medium', 'high'])
    df['ph_category_encoded'] = LabelEncoder().fit_transform(df['ph_category'].astype(str))
    
    # Список признаков (полностью совпадает с ноутбуком)
    feature_cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
        'wine_type_encoded', 'free_to_total_sulfur_ratio', 'acid_balance', 'sugar_alcohol_ratio',
        'ph_category_encoded'
    ]
    X = df[feature_cols]
    y = df['quality_binary']
    
    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    return X_scaled, y, scaler, feature_cols, df

X, y, scaler, feature_cols, df = load_and_preprocess()

# Боковая панель: выбор модели и гиперпараметров
st.sidebar.header("⚙️ Настройки модели")

model_choice = st.sidebar.selectbox(
    "Выберите модель:",
    ["Random Forest", "Gradient Boosting", "Decision Tree",
     "K-Nearest Neighbors", "SVM", "Logistic Regression", "AdaBoost"]
)

st.sidebar.subheader("Гиперпараметры")

def get_model(name, params):
    if name == "Random Forest":
        return RandomForestClassifier(**params, random_state=42)
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(**params, random_state=42)
    elif name == "Decision Tree":
        return DecisionTreeClassifier(**params, random_state=42)
    elif name == "K-Nearest Neighbors":
        return KNeighborsClassifier(**params)
    elif name == "SVM":
        return SVC(**params, probability=True, random_state=42)
    elif name == "Logistic Regression":
        return LogisticRegression(**params, random_state=42)
    else:  # AdaBoost
        if 'estimator' in params:
            if params['estimator'] == 'depth1':
                params['estimator'] = DecisionTreeClassifier(max_depth=1)
            elif params['estimator'] == 'depth3':
                params['estimator'] = DecisionTreeClassifier(max_depth=3)
            else:
                del params['estimator']
        return AdaBoostClassifier(**params, random_state=42)

params = {}
if model_choice == "Random Forest":
    params['n_estimators'] = st.sidebar.slider("Количество деревьев", 10, 300, 100, 10)
    params['max_depth'] = st.sidebar.slider("Максимальная глубина", 2, 30, 10, 1)
    params['min_samples_split'] = st.sidebar.slider("Мин. образцов для разделения", 2, 20, 2, 1)
    params['min_samples_leaf'] = st.sidebar.slider("Мин. образцов в листе", 1, 10, 1, 1)
elif model_choice == "Gradient Boosting":
    params['n_estimators'] = st.sidebar.slider("Количество деревьев", 10, 300, 100, 10)
    params['learning_rate'] = st.sidebar.slider("Скорость обучения", 0.01, 1.0, 0.1, 0.01)
    params['max_depth'] = st.sidebar.slider("Максимальная глубина", 2, 15, 3, 1)
    params['min_samples_split'] = st.sidebar.slider("Мин. образцов для разделения", 2, 20, 2, 1)
elif model_choice == "Decision Tree":
    params['max_depth'] = st.sidebar.slider("Максимальная глубина", 2, 30, 10, 1)
    params['min_samples_split'] = st.sidebar.slider("Мин. образцов для разделения", 2, 20, 2, 1)
    params['min_samples_leaf'] = st.sidebar.slider("Мин. образцов в листе", 1, 10, 1, 1)
    params['criterion'] = st.sidebar.selectbox("Критерий", ["gini", "entropy"])
elif model_choice == "K-Nearest Neighbors":
    params['n_neighbors'] = st.sidebar.slider("Количество соседей", 1, 20, 5, 1)
    params['weights'] = st.sidebar.selectbox("Веса", ["uniform", "distance"])
    params['metric'] = st.sidebar.selectbox("Метрика", ["euclidean", "manhattan"])
elif model_choice == "SVM":
    params['C'] = st.sidebar.slider("C", 0.1, 100.0, 1.0, 0.1)
    params['kernel'] = st.sidebar.selectbox("Ядро", ["rbf", "linear", "poly", "sigmoid"])
    params['gamma'] = st.sidebar.selectbox("Gamma", ["scale", "auto"])
elif model_choice == "Logistic Regression":
    params['C'] = st.sidebar.slider("C", 0.01, 100.0, 1.0, 0.01)
    params['max_iter'] = st.sidebar.slider("Макс. итераций", 100, 2000, 1000, 100)
    params['solver'] = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear"])
else:  # AdaBoost
    params['n_estimators'] = st.sidebar.slider("Количество оценщиков", 50, 300, 100, 50)
    params['learning_rate'] = st.sidebar.slider("Скорость обучения", 0.01, 1.0, 0.1, 0.01)
    params['estimator'] = st.sidebar.selectbox("Базовый классификатор", ["depth1", "depth3"])

# Создание и обучение модели
model = get_model(model_choice, params)
model.fit(X, y)  # обучаем на всех данных для демо

# Вкладки приложения
tab1, tab2, tab3 = st.tabs(["🔬 Предсказание", "📊 Оценка модели", "📈 Данные"])

with tab1:
    st.header("Введите характеристики вина")
    col1, col2, col3 = st.columns(3)
    with col1:
        fixed_acidity = st.slider("Фиксированная кислотность", 4.0, 16.0, 7.0, 0.1)
        volatile_acidity = st.slider("Летучая кислотность", 0.1, 1.6, 0.5, 0.01)
        citric_acid = st.slider("Лимонная кислота", 0.0, 1.0, 0.3, 0.01)
        pH = st.slider("pH", 2.7, 4.0, 3.3, 0.01)
    with col2:
        residual_sugar = st.slider("Остаточный сахар", 0.6, 66.0, 5.0, 0.1)
        alcohol = st.slider("Алкоголь (%)", 8.0, 15.0, 10.5, 0.1)
        density = st.slider("Плотность", 0.987, 1.004, 0.997, 0.001)
    with col3:
        free_sulfur_dioxide = st.slider("Свободный SO2", 1.0, 289.0, 30.0, 1.0)
        total_sulfur_dioxide = st.slider("Общий SO2", 6.0, 440.0, 120.0, 1.0)
        chlorides = st.slider("Хлориды", 0.01, 0.6, 0.08, 0.001)
        sulphates = st.slider("Сульфаты", 0.2, 2.0, 0.5, 0.01)
    wine_type = st.selectbox("Тип вина", ["Красное", "Белое"])
    wine_type_encoded = 1 if wine_type == "Белое" else 0

    # Расчет производных признаков
    free_to_total_sulfur_ratio = free_sulfur_dioxide / (total_sulfur_dioxide + 1e-6)
    acid_balance = fixed_acidity / (volatile_acidity + 1e-6)
    sugar_alcohol_ratio = residual_sugar / (alcohol + 1e-6)
    if pH < 3.0:
        ph_category_encoded = 0
    elif pH < 3.5:
        ph_category_encoded = 1
    else:
        ph_category_encoded = 2

    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol],
        'wine_type_encoded': [wine_type_encoded],
        'free_to_total_sulfur_ratio': [free_to_total_sulfur_ratio],
        'acid_balance': [acid_balance],
        'sugar_alcohol_ratio': [sugar_alcohol_ratio],
        'ph_category_encoded': [ph_category_encoded]
    })
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        prob_good = proba[1] if len(proba) > 1 else 0.0
    else:
        prob_good = None

    st.markdown("---")
    st.subheader("Результат")
    colr1, colr2 = st.columns(2)
    with colr1:
        st.markdown("### Предсказанное качество:")
        if prediction == 1:
            st.markdown('<div class="prediction-good">⭐ Хорошее вино (quality ≥ 6)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-bad">👎 Плохое вино (quality < 6)</div>', unsafe_allow_html=True)
        st.markdown(f"**Тип вина:** {wine_type}")
        st.markdown(f"**Модель:** {model_choice}")
    with colr2:
        if prob_good is not None:
            st.markdown("### Вероятность того, что вино хорошее:")
            fig, ax = plt.subplots()
            ax.bar(["Плохое", "Хорошее"], [1-prob_good, prob_good], color=['#e74c3c', '#27ae60'])
            ax.set_ylabel('Вероятность')
            st.pyplot(fig)

with tab2:
    st.header("Оценка модели (кросс-валидация)")
    with st.spinner("Вычисляется кросс-валидация..."):
        # Используем 5-фолдовую стратифицированную кросс-валидацию на ВСЕХ данных
        scorer = make_scorer(f1_score)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer, n_jobs=-1)
        st.write(f"Средний F1‑score (5‑fold CV): **{cv_scores.mean():.4f}** ± {cv_scores.std():.4f}")
        st.write(f"Accuracy на всех данных (может быть завышена): {accuracy_score(y, model.predict(X)):.4f}")
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Важность признаков")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Признак': feature_cols, 'Важность': importances}).sort_values('Важность', ascending=True)
        fig, ax = plt.subplots(figsize=(10,8))
        sns.barplot(data=imp_df, x='Важность', y='Признак', palette='viridis', ax=ax)
        ax.set_title(f'Важность признаков — {model_choice}')
        st.pyplot(fig)

with tab3:
    st.header("Датасет Wine Quality")
    st.markdown("""
    Полный датасет содержит 6497 образцов красного и белого вина с химическими характеристиками.
    В задаче используется бинарная целевая переменная: **хорошее вино (quality ≥ 6)**.
    """)
    st.dataframe(df.head(10))
    st.subheader("Статистика")
    st.dataframe(df.describe().round(3))