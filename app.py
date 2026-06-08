from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Qualidade de Vinhos", page_icon="🍷", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


@st.cache_resource
def load_models():
    pipeline_path = MODEL_DIR / "pipeline_vinhos_final.pkl"

    if pipeline_path.exists():
        return joblib.load(pipeline_path), None

    modelo = joblib.load(MODEL_DIR / "modelo_vinhos_final.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler_vinhos.pkl")
    return modelo, scaler


modelo, scaler = load_models()

st.title("🍷 Classificador de Vinhos Tintos")
st.markdown("Insira as propriedades físico-químicas do vinho para prever sua qualidade.")

st.subheader("Propriedades Químicas")

col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input("Teor Alcoólico", min_value=8.0, max_value=15.0, value=10.0, step=0.1)
    sulphates = st.number_input("Sulfatos", min_value=0.0, max_value=2.0, value=0.6, step=0.01)
    volatile_acidity = st.number_input("Acidez Volátil", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    citric_acid = st.number_input("Ácido Cítrico", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    residual_sugar = st.number_input("Açúcar Residual", min_value=0.0, max_value=16.0, value=2.0, step=0.1)
    chlorides = st.number_input("Cloretos", min_value=0.0, max_value=0.7, value=0.08, step=0.001)

with col2:
    free_sulfur_dioxide = st.number_input("Dióxido de Enxofre Livre", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    total_sulfur_dioxide = st.number_input("Dióxido de Enxofre Total", min_value=0.0, max_value=300.0, value=45.0, step=1.0)
    density = st.number_input("Densidade", min_value=0.990, max_value=1.005, value=0.996, step=0.0001)
    pH = st.number_input("pH", min_value=2.5, max_value=4.5, value=3.3, step=0.01)
    fixed_acidity = st.number_input("Acidez Fixa", min_value=4.0, max_value=16.0, value=8.0, step=0.1)

if st.button("Analisar Qualidade", type="primary"):
    entrada = pd.DataFrame(
        [[
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol,
        ]],
        columns=FEATURE_NAMES,
    )

    dados_modelo = entrada if scaler is None else scaler.transform(entrada)
    predicao = modelo.predict(dados_modelo)[0]
    probabilidade = modelo.predict_proba(dados_modelo)[0, 1]

    st.divider()
    if predicao == 1:
        st.success("🌟 O modelo previu que este é um **Vinho Bom** (Nota >= 7)!")
    else:
        st.warning("⚠️ O modelo previu que este é um **Vinho Ruim/Médio** (Nota < 7).")

    st.metric("Probabilidade estimada de vinho bom", f"{probabilidade:.1%}")
    st.caption(
        "A previsão é uma estimativa baseada nas propriedades químicas informadas "
        "e não substitui uma avaliação sensorial especializada."
    )
