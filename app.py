import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool

# =========================
#  Carga de modelo y datos
# =========================
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_cars.cbm")
    return model

@st.cache_data
def load_data():
    cars = pd.read_csv("cars_clean.csv")

    # Si falta mileage_km pero existe mileage, lo creamos sobre la marcha
    if "mileage_km" not in cars.columns and "mileage" in cars.columns:
        MILES_TO_KM = 1.60934
        cars["mileage_km"] = cars["mileage"] * MILES_TO_KM

    # Si falta consumo pero existe mpg, lo creamos
    if "consumption_l_100km" not in cars.columns and "mpg" in cars.columns:
        IMP_MPG_TO_L_100KM = 282.481
        cars["consumption_l_100km"] = IMP_MPG_TO_L_100KM / cars["mpg"]

    return cars


model = load_model()
cars = load_data()

CAT_FEATURES = ["brand", "model", "transmission", "fuelType"]

# Métricas del modelo (de tu validación)
MAE_MODEL = 1398
R2_MODEL = 0.9563

# =========================
#  Configuración de página
# =========================
st.set_page_config(
    page_title="Predicción precio de coches",
    page_icon="🚗",
    layout="wide",
)

# Un poco de CSS para darle estilo
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #111827, #020617);
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background-color: #020617;
    }
    .stMetric > div {
        background-color: #0f172a !important;
        border-radius: 0.75rem;
        padding: 0.75rem;
    }
    .result-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 1.5rem 1.75rem;
        border-radius: 1rem;
        border: 1px solid #4b5563;
    }
    .chip {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background-color: #111827;
        color: #e5e7eb;
        font-size: 0.75rem;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
        border: 1px solid #4b5563;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
#  Sidebar
# =========================
st.sidebar.title("⚙️ Modelo CatBoost")

st.sidebar.markdown("### Rendimiento")
col_mae, col_r2 = st.sidebar.columns(2)
with col_mae:
    st.metric("MAE", f"{MAE_MODEL:,.0f} €")
with col_r2:
    st.metric("R²", f"{R2_MODEL:.3f}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Datos usados")

st.sidebar.write(f"**{len(cars):,} coches**")
st.sidebar.write(f"**{cars['brand'].nunique()} marcas**")
st.sidebar.write(f"**{cars['model'].nunique()} modelos**")

st.sidebar.markdown("---")
st.sidebar.caption(
    "El modelo se ha entrenado con datos reales, usando CatBoost para manejar "
    "variables categóricas (marca, modelo, transmisión y combustible)."
)

# =========================
#  Cabecera principal
# =========================
st.markdown(
    """
    <h1 style="margin-bottom:0.2rem;">🚗 Estimador de precio de coches</h1>
    <p style="color:#9ca3af; font-size:0.95rem; margin-bottom:0.5rem;">
        Introduce las características del vehículo y el modelo te devuelve una estimación de su precio de mercado.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================
#  Inputs del usuario
# =========================
st.subheader("🔧 Configura tu coche")

col1, col2 = st.columns(2)

with col1:
    # Marca y modelo
    brand_options = sorted(cars["brand"].unique())
    brand = st.selectbox("Marca", brand_options)

    model_options = sorted(cars[cars["brand"] == brand]["model"].unique())
    modelo = st.selectbox("Modelo", model_options)

    years = [y for y in sorted(cars["year"].unique()) if 1995 <= y <= 2020]
    year = st.selectbox("Año", years, index=len(years) - 1)

with col2:
    engine_sizes = sorted(cars["engineSize"].unique())
    engineSize = st.selectbox("Motor - cilindrada (L)", engine_sizes)

    avg_mileage = int(cars["mileage_km"].median())
    mileage_km = st.number_input(
        "Kilómetros",
        min_value=0,
        max_value=500_000,
        value=avg_mileage,
        step=5000,
    )

    avg_consumption = float(cars["consumption_l_100km"].median())
    consumption_l_100km = st.number_input(
        "Consumo (L/100 km)",
        min_value=2.0,
        max_value=20.0,
        value=round(avg_consumption, 1),
        step=0.1,
    )

col3, col4 = st.columns(2)
with col3:
    transmissions = sorted(cars["transmission"].unique())
    transmission = st.selectbox("Transmisión", transmissions)

with col4:
    fuel_display = {
        "Petrol": "Gasolina",
        "Diesel": "Diésel",
        "Electric": "Eléctrico",
        "Hybrid": "Híbrido",
        "Other": "Otro",
    }
    fuel_options_display = [fuel_display[f] for f in sorted(cars["fuelType"].unique())]
    fuel_human = st.selectbox("Combustible", fuel_options_display)
    fuelType = [k for k, v in fuel_display.items() if v == fuel_human][0]

st.markdown("")

# =========================
#  Predicción
# =========================
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("🔍 Predecir precio", use_container_width=True)

if predict_clicked:
    # Edad del coche (feature derivada usada por el modelo)
    base_year = 2020
    car_age = base_year - year

    data = {
        "model": [modelo],
        "year": [year],
        "car_age": [car_age],
        "mileage_km": [mileage_km],
        "engineSize": [engineSize],
        "consumption_l_100km": [consumption_l_100km],
        "brand": [brand],
        "transmission": [transmission],
        "fuelType": [fuelType],
    }

    df_input = pd.DataFrame(data)
    pool_input = Pool(df_input, cat_features=CAT_FEATURES)
    pred_price = float(model.predict(pool_input)[0])

    min_price = pred_price - MAE_MODEL
    max_price = pred_price + MAE_MODEL

    st.markdown("## 🧾 Resultado de la estimación")

    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="margin-top:0;">💰 Precio estimado</h3>
            <p style="font-size:2rem; font-weight:700; margin-bottom:0.25rem;">
                {pred_price:,.0f} € 
            </p>
            <p style="color:#9ca3af; margin-bottom:0.5rem;">
                Intervalo aproximado (± MAE ≈ {MAE_MODEL:,.0f} €):
                <br>
                <span style="font-weight:600;">{min_price:,.0f} €</span>
                — 
                <span style="font-weight:600;">{max_price:,.0f} €</span>
            </p>
            <p style="color:#9ca3af; font-size:0.85rem; margin-bottom:0.75rem;">
                El intervalo representa el rango en el que normalmente se mueve el error del modelo
                según su rendimiento en datos reales de prueba.
            </p>
            <div>
                <span class="chip">Marca: {brand}</span>
                <span class="chip">Modelo: {modelo}</span>
                <span class="chip">Año: {year}</span>
                <span class="chip">Km: {mileage_km:,}</span>
                <span class="chip">Motor: {engineSize} L</span>
                <span class="chip">Consumo: {consumption_l_100km:.1f} L/100 km</span>
                <span class="chip">Transmisión: {transmission}</span>
                <span class="chip">Combustible: {fuel_human}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

