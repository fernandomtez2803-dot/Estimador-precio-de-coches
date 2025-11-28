import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
import plotly.express as px


#  Carga de modelo y datos

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

@st.cache_data
def compute_depreciation_table(cars: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula una pendiente precio~año por modelo.
    Pendiente = cambio medio de precio por año (€/año).
    Cuanto más ALTA es la pendiente, mejor mantiene su valor.
    """
    rows = []
    tmp = cars[['brand', 'model', 'year', 'price']].dropna()

    for (brand, model), sub in tmp.groupby(['brand', 'model']):
        # Si no hay años suficientes o pocas observaciones, no usamos ese modelo
        if sub['year'].nunique() < 3 or len(sub) < 30:
            continue
        try:
            coef = np.polyfit(sub['year'], sub['price'], 1)[0]  # €/año
            rows.append([brand, model, coef])
        except Exception:
            pass

    df = pd.DataFrame(rows, columns=['brand', 'model', 'slope'])
    # Ordenamos de mayor a menor pendiente (mejor inversión primero)
    df = df.sort_values('slope', ascending=False).reset_index(drop=True)
    return df


def get_model_ranks(brand: str, model: str, depr_table: pd.DataFrame):
    """
    Devuelve info de ranking para un modelo:
    - slope: pendiente €/año
    - pos_best: posición en ranking de mejores inversiones (1 = mejor)
    - pos_worst: posición en ranking de los que más se deprecian (1 = se deprecia más)
    - total: número total de modelos
    """
    if depr_table.empty:
        return None

    mask = (depr_table["brand"] == brand) & (depr_table["model"] == model)
    if not mask.any():
        return None

    slope_value = float(depr_table.loc[mask, "slope"].iloc[0])

    ranking_best = depr_table.sort_values("slope", ascending=False).reset_index(drop=True)
    ranking_worst = depr_table.sort_values("slope", ascending=True).reset_index(drop=True)

    idx_best = ranking_best.index[
        (ranking_best["brand"] == brand) & (ranking_best["model"] == model)
    ][0]
    idx_worst = ranking_worst.index[
        (ranking_worst["brand"] == brand) & (ranking_worst["model"] == model)
    ][0]

    total = len(depr_table)

    return {
        "slope": slope_value,
        "pos_best": int(idx_best) + 1,
        "pos_worst": int(idx_worst) + 1,
        "total": int(total),
    }


model = load_model()
cars = load_data()
depr_table = compute_depreciation_table(cars)

CAT_FEATURES = ["brand", "model", "transmission", "fuelType"]

# Métricas del modelo (de tu validación)
MAE_MODEL = 1398
R2_MODEL = 0.9563

# Configuración de página
st.set_page_config(
    page_title="Predicción precio de coches",
    page_icon="🚗",
    layout="wide",
)

# CSS para darle estilo
st.markdown("---")

st.markdown(
    """
    <h1 style='text-align: center; color: #ffffff; margin-bottom: 5px;'>
         Modelo avanzado de valoración y rentabilidad de vehículos
    </h1>
    <p style='text-align: center; color: #b5b5b5; font-size: 17px; margin-top: -10px;'>
        Datos reales
    </p>
    """,
    unsafe_allow_html=True
)


#  Pestañas principales

tab_pred, tab_inv = st.tabs(["🔮 Predicción", "💼 Inversión"])


#  TAB 1: PREDICCIÓN

with tab_pred:

    # Inputs del usuario
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

    #  Consumo (L/100 km)
    subset = cars.copy()

    # Nivel 1: misma marca, modelo, motor y combustible
    mask_full = (
        (subset["brand"] == brand) &
        (subset["model"] == modelo) &
        (subset["engineSize"] == engineSize) &
        (subset["fuelType"] == fuelType)
    )
    cand = subset[mask_full]

    # Si hay pocos, relajamos la condición poco a poco
    if len(cand) < 5:
        mask_brand_model_fuel = (
            (subset["brand"] == brand) &
            (subset["model"] == modelo) &
            (subset["fuelType"] == fuelType)
        )
        cand = subset[mask_brand_model_fuel]

    if len(cand) < 5:
        mask_brand_fuel = (
            (subset["brand"] == brand) &
            (subset["fuelType"] == fuelType)
        )
        cand = subset[mask_brand_fuel]

    if len(cand) < 5:
        mask_fuel = subset["fuelType"] == fuelType
        cand = subset[mask_fuel]

    # Si sigue habiendo muy pocos, usamos la mediana global
    if len(cand) > 0:
        default_consumption = float(cand["consumption_l_100km"].median())
    else:
        default_consumption = float(subset["consumption_l_100km"].median())

    use_auto_consumption = st.checkbox(
        "No me sé el consumo (usar valor aproximado)",
        value=False
    )

    if not use_auto_consumption:
        consumption_l_100km = st.number_input(
            "Consumo (L/100 km)",
            min_value=2.0,
            max_value=20.0,
            value=round(default_consumption, 1),
            step=0.1,
            help="Si no lo sabes, marca la casilla de arriba y usa este valor aproximado.",
        )
    else:
        st.info(
            f"Usando un consumo típico de **{default_consumption:.1f} L/100 km** "
            f"para un {brand} {modelo} con este motor y combustible."
        )
        consumption_l_100km = default_consumption

    #  Predicción
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔍 Predecir precio", use_container_width=True)

    if predict_clicked:
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

        st.markdown("##  Resultado de la estimación")
        st.markdown(
            f"""
            <div class="result-card">
                <h3 style="margin-top:0;">💰 Precio estimado</h3>
                <p style="font-size:2rem; font-weight:700; margin-bottom:0.25rem;">
                    {pred_price:,.0f} € 
                </p>
                <p style="color:#9ca3af; margin-bottom:0.5rem;">
                    Intervalo aproximado :
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

        # Gráficos dinámicos post-predicción
        st.markdown("### 📈 Cómo cambia el precio para coches como este")
        tab_km, tab_year = st.tabs(["Precio vs km", "Precio vs año"])

        # Precio vs km 
        with tab_km:
            km_min = max(0, mileage_km - 100_000)
            km_max = mileage_km + 100_000
            if km_max == km_min:
                km_max = km_min + 100_000

            km_values = np.linspace(km_min, km_max, 20, dtype=int)

            df_km = pd.DataFrame({
                "model": [modelo] * len(km_values),
                "year": [year] * len(km_values),
                "car_age": [base_year - year] * len(km_values),
                "mileage_km": km_values,
                "engineSize": [engineSize] * len(km_values),
                "consumption_l_100km": [consumption_l_100km] * len(km_values),
                "brand": [brand] * len(km_values),
                "transmission": [transmission] * len(km_values),
                "fuelType": [fuelType] * len(km_values),
            })

            pool_km = Pool(df_km, cat_features=CAT_FEATURES)
            df_km["Precio estimado (€)"] = model.predict(pool_km)

            fig_km = px.line(
                df_km,
                x="mileage_km",
                y="Precio estimado (€)",
                markers=True,
                labels={"mileage_km": "Kilómetros", "Precio estimado (€)": "Precio estimado (€)"},
                title="Impacto del kilometraje en el precio estimado",
            )
            # Línea vertical para marcar los km que has metido
            fig_km.add_vline(
                x=mileage_km,
                line_dash="dash",
                line_color="#f97316",
                annotation_text="Km introducidos",
                annotation_position="top left",
            )

            # Punto destacado de tu coche (en esos km)
            fig_km.add_scatter(
                x=[mileage_km],
                y=[pred_price],
                mode="markers",
                marker=dict(size=9),
                name="Tu coche",
            )
            fig_km.update_layout(
                template="plotly_dark",
                height=420,
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig_km, use_container_width=True)
            st.caption("Simulación manteniendo fijo el resto de características y variando solo los kilómetros.")

        #  Precio vs año
        #       
        with tab_year:
            # Año base del modelo 
            base_year = 2020
            years_available = sorted(y for y in cars["year"].unique() if 1995 <= y <= base_year)

            # Tomamos hasta 3 años hacia atrás desde el año de tu coche (misma lógica del modelo)
            min_year_range = max(min(years_available), year - 3)
            years_range = [y for y in years_available if min_year_range <= y <= base_year]

            # Si por lo que sea hay muy pocos, usamos todos como fallback
            if len(years_range) < 3:
                years_range = years_available

            df_year = pd.DataFrame({
                "model": [modelo] * len(years_range),
                "year": years_range,
                "mileage_km": [mileage_km] * len(years_range),
                "engineSize": [engineSize] * len(years_range),
                "consumption_l_100km": [consumption_l_100km] * len(years_range),
                "brand": [brand] * len(years_range),
                "transmission": [transmission] * len(years_range),
                "fuelType": [fuelType] * len(years_range),
            })
            df_year["car_age"] = base_year - df_year["year"]

            pool_year = Pool(df_year, cat_features=CAT_FEATURES)
            df_year["Precio estimado (€)"] = model.predict(pool_year)

            # Forzamos que el punto correspondiente al coche que has configurado
            # coincida EXACTAMENTE con la predicción principal
            df_year.loc[df_year["year"] == year, "Precio estimado (€)"] = pred_price

            # ==== REESCALAMOS EL EJE X PARA QUE TU PUNTO SEA 2025 ====
            current_year_display = 2025  # Año actual (cuando haces la predicción)
            # Este será el eje "de mentira" para visualizar: 2015, 2016, ..., 2025
            df_year["año_escenario"] = df_year["year"] - year + current_year_display

           
                        # === EXTENDER LA CURVA HASTA 2030 (5 años más) ===
            target_max_year = 2030  # año máximo que queremos ver en el gráfico

            # Ordenamos por año de escenario por si acaso
            df_year = df_year.sort_values("año_escenario")

            # DataFrame que usaremos para pintar (lo podremos alargar)
            df_plot = df_year[["año_escenario", "Precio estimado (€)"]].copy()

            max_escenario = df_plot["año_escenario"].max()

            # Solo extrapolamos si aún no llegamos a 2030 y tenemos al menos 2 puntos
            if max_escenario < target_max_year and len(df_plot) >= 2:
                # Pendiente (€/año) entre los dos últimos puntos
                last_two = df_plot.tail(2)
                dy = last_two.iloc[1]["Precio estimado (€)"] - last_two.iloc[0]["Precio estimado (€)"]
                dx = last_two.iloc[1]["año_escenario"] - last_two.iloc[0]["año_escenario"]
                slope = dy / dx if dx != 0 else 0

                last_year = max_escenario
                last_price = last_two.iloc[1]["Precio estimado (€)"]

                # Vamos añadiendo años hasta 2030 siguiendo esa pendiente
                for new_year in range(int(last_year) + 1, target_max_year + 1):
                    last_price += slope
                    df_plot.loc[len(df_plot)] = [new_year, last_price]

            # A partir de aquí, usamos df_plot en vez de df_year para el gráfico
            fig_year = px.line(
                df_plot,
                x="año_escenario",
                y="Precio estimado (€)",
                markers=True,
                labels={
                    "año_escenario": "Año (escenario)",
                    "Precio estimado (€)": "Precio estimado (€)",
                },
                title="Cómo habría cambiado el precio de un coche igual",
            )


            # Línea vertical para marcar claramente el año actual (predicción)
            fig_year.add_vline(
                x=current_year_display,
                line_dash="dash",
                line_color="#f97316",
                annotation_text="Año actual (2025)",
                annotation_position="top left",
            )

            fig_year.update_layout(
                template="plotly_dark",
                height=520,
                margin=dict(l=20, r=20, t=80, b=60),
                title_font_size=22,
                xaxis_title_font_size=16,
                yaxis_title_font_size=16,
                xaxis=dict(tickmode="linear", dtick=1),
            )

            st.plotly_chart(fig_year, use_container_width=True)
            st.caption(
                "Tomamos tu coche tal y como lo has configurado y usamos el modelo para estimar "
                "qué habría valido si fuera más antiguo o más nuevo, manteniendo el mismo kilometraje. "
                "El punto de 2025 es el precio predicho actual."
            )
 
             
        


#  TAB 2: INVERSIÓN

with tab_inv:

    #  Ranking de inversión de TU coche
    st.markdown("###  Tu coche en el ranking de inversión")

    ranks = get_model_ranks(brand, modelo, depr_table)

    if ranks is None:
        st.caption(
            "El modelo seleccionado no aparece en la tabla de depreciación "
            "(no hay datos históricos suficientes para calcular su ranking)."
        )
    else:
        slope_value = ranks["slope"]
        pos_best = ranks["pos_best"]
        total_models = ranks["total"]

        # Resumen del modelo
        mask_price_model = (cars["brand"] == brand) & (cars["model"] == modelo)
        avg_price_model = cars.loc[mask_price_model, "price"].mean()

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(
                "Posición en ranking (1 = mejor inversión)",
                f"{pos_best} / {total_models}",
            )

        with col_r2:
            if not np.isnan(avg_price_model) and avg_price_model != 0:
                perc_year = slope_value / avg_price_model * 100
                sign = "+" if perc_year >= 0 else ""
                delta_str = f"{sign}{perc_year:.1f}% / año"

                st.metric(
                    "Variación anual del precio",
                    f"{slope_value:,.0f} €/año",
                    delta=delta_str,
                    help=(
                        "Cambio medio del precio cada año según los datos del mercado. "
                        "Si es negativo, el modelo pierde valor; si está cerca de 0, mantiene bien el precio."
                    ),
                )
            else:
                st.metric(
                    "Variación anual del precio",
                    f"{slope_value:,.0f} €/año",
                )

        # Modelos más parecidos (vecinos)
        ranking_sorted = depr_table.sort_values("slope", ascending=False).reset_index(drop=True)
        ranking_sorted["pos_inversion"] = ranking_sorted.index + 1

        mask_model = (ranking_sorted["brand"] == brand) & (ranking_sorted["model"] == modelo)

        if mask_model.any():
            idx = ranking_sorted.index[mask_model][0]
            start = max(0, idx - 5)
            end = min(len(ranking_sorted), idx + 6)
            view = ranking_sorted.iloc[start:end].copy()
        else:
            view = ranking_sorted.head(5).copy()

        view["tu_modelo"] = np.where(
            (view["brand"] == brand) & (view["model"] == modelo),
            "⭐",
            ""
        )

        view["variación_€/año"] = view["slope"].round(0).astype(int)

        st.markdown("#### Modelos con comportamiento similar al tuyo")
        st.dataframe(
            view[["pos_inversion", "brand", "model", "variación_€/año", "tu_modelo"]],
            use_container_width=True,
            hide_index=True,
        )

    # Explicación general y rankings globales
    if depr_table.empty:
        st.info(
            "Todavía no se ha podido calcular la depreciación por modelo. "
            "Revisa que el dataset tenga suficientes años y observaciones por modelo."
        )
    else:
        st.markdown(
            """
           
            **Modelos que mejor mantienen su valor con el paso del tiempo**

            En esta sección se muestran los modelos de coche que, según los datos históricos del mercado, conservan mejor su precio a lo largo de los años.

            Estos modelos destacan porque su valor tiende a depreciarse menos e incluso, en algunos casos, a subir debido a la demanda, rareza o características especiales.

            La tabla ordena los vehículos de mayor a menor rendimiento en este sentido, permitiendo identificar fácilmente qué coches son una mejor inversión a largo plazo.

            
            """
        )

        tab_modelos, tab_marcas = st.tabs(["Ranking por modelo", "Ranking por marca"])

        # Ranking por modelo 
        with tab_modelos:
            top_n_models = st.slider(
                "Número de modelos a mostrar",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                key="slider_models",
            )

            ranking_best = depr_table.sort_values("slope", ascending=False).reset_index(drop=True)
            top_models = ranking_best.head(top_n_models).copy()

            def interpretar_pendiente(x):
                if x > 0:
                    return "En datos sube de precio"
                elif x > -800:
                    return "Se deprecia muy poco"
                elif x > -2000:
                    return "Depreciación media"
                else:
                    return "Se deprecia rápido"

            top_models["pendiente_€/año"] = top_models["slope"].round(0).astype(int)
            top_models["interpretación"] = top_models["slope"].apply(interpretar_pendiente)

            st.markdown("#### 🏆 Modelos que mejor mantienen su valor")
            st.dataframe(
                top_models[["brand", "model", "pendiente_€/año", "interpretación"]],
                use_container_width=True,
                hide_index=True,
            )
# Ordenamos de mayor a menor pendiente
            df_models_plot = top_models.sort_values("slope", ascending=True).head(top_n_models)

            fig_models = px.bar(
                df_models_plot,
                x="pendiente_€/año",
                y="model",
                color="brand",
                orientation="h",
                labels={
                    "pendiente_€/año": "Pendiente (€/año, más alta = mejor inversión)",
                    "model": "Modelo",
                },
                title="Modelos que se deprecian más lento",
       
)

            fig_models.update_layout(
                template="seaborn",
                height=520,
                margin=dict(l=0, r=10, t=50, b=10),
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=df_models_plot["model"].tolist()
    ),
)

            st.plotly_chart(fig_models, use_container_width=True)

        # Ranking por marca 
        with tab_marcas:
            top_n_brands = st.slider(
                "Número de marcas a mostrar",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key="slider_brands",
            )

            brand_depr = (
                depr_table
                .groupby("brand", as_index=False)["slope"]
                .mean()
                .sort_values("slope", ascending=False)
                .reset_index(drop=True)
            )

            brand_depr["pendiente_€/año"] = brand_depr["slope"].round(0).astype(int)

            st.markdown("#### 🏅 Marcas que mejor mantienen su valor")
            st.dataframe(
                brand_depr.head(top_n_brands)[["brand", "pendiente_€/año"]],
                use_container_width=True,
                hide_index=True,
            )


            

            fig_brands = px.bar(
                brand_depr.head(top_n_brands).sort_values("slope", ascending=True),
                x="pendiente_€/año",
                y="brand",
                orientation="h",
                labels={
                    "pendiente_€/año": "Ganancia media (€/año)",
                    "brand": "Marca",
                },
                title="Marcas que mejor mantienen su valor",
            )
            fig_brands.update_layout(
                template="seaborn",
                height=520,
                margin=dict(l=0, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_brands, use_container_width=True)

        
