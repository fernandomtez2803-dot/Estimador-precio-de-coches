🚗 Predicción de precios de coches + Análisis de depreciación
Proyecto de Machine Learning con CatBoost + App interactiva en Streamlit
📌 1. Objetivo del proyecto

El objetivo principal es construir un modelo de Machine Learning robusto capaz de:

Predecir el precio realista de un coche a partir de sus características.

Analizar su depreciación anual en función del mercado (pendiente €/año).

Posicionarlo dentro de un ranking de inversión, comparándolo con coches similares.

Explorar visualmente cómo cambia el precio según:

los kilómetros

el año de matriculación

Todo está integrado en una app de Streamlit totalmente interactiva.

📦 2. Dataset y características usadas

El dataset incluye miles de coches reales con las siguientes variables:

Tipo	Variable
Numéricas	price_eur, year, mileage_km, engineSize, consumption_l_100km
Categóricas	brand, model, transmission, fuelType

Estas features fueron seleccionadas porque mantienen una relación directa y demostrable con el precio de mercado.

🧼 3. Limpieza y preprocesado

Se aplicó un proceso de limpieza estándar:

✔ Eliminación de valores extremos

Se filtraron:

años fuera de rango (1995–2020),

precios obviamente incorrectos,

motores o consumos irreales.

✔ Conversión de tipos

Las variables categóricas se mantuvieron como string para aprovechar CatBoost, que las procesa de forma nativa.

✔ Feature Engineering

Se añadió:

car_age = año_base - year
Esta variable acelera la comprensión del modelo sobre depreciación.

Estandarización no necesaria (CatBoost no lo requiere).

📊 4. Análisis exploratorio (EDA)
🔥 Matriz de correlación

La correlación mostró relaciones clave:

Año ↗ correlaciona fuerte con precio (0.50)
→ coches más nuevos valen más.

Kilómetros ↘ correlación negativa (-0.43)
→ más uso, menos precio.

El engineSize muestra una correlación media (0.63)
→ motores más grandes suelen venderse más caros.

Consumo tiene un impacto más débil.

Esta matriz permitió identificar las features más relevantes para el modelo.

🧠 5. Modelo de Machine Learning: CatBoost

Se decidió usar CatBoostRegressor porque:

✔ Maneja datos categóricos sin necesidad de One-Hot Encoding

XGBoost y RandomForest requieren transformar variables categóricas → más dimensiones, más tiempo, más riesgo de overfitting.
CatBoost trabaja directamente con string categories mediante target encoding ordenado.

✔ Robusto, rápido y con excelente rendimiento en datasets tabulares

En competiciones de Kaggle, CatBoost suele superar a RF y GB tradicionales.

✔ Permite un entrenamiento estable incluso con datos ruidosos

Ideal para datos reales de mercado automovilístico.

⚙ 6. Parámetros del modelo
model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    n_estimators=2000,
    l2_leaf_reg=5,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=42,
    verbose=200
)

Explicación técnica breve:

depth=8 → complejidad de los árboles

learning_rate=0.05 → aprendizaje gradual, reduce overfitting

n_estimators=2000 → más iteraciones = mejor ajuste

l2_leaf_reg=5 → regularización L2 para estabilidad

MAE como métrica → más robusta que MSE ante outliers

verbose=200 → logging durante el entrenamiento

🧩 7. ¿Qué es un Pool en CatBoost?

CatBoost usa estructuras internas llamadas Pool:

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_features)


Un Pool indica:

qué columnas son categóricas

cómo deben ser tratadas en el pipeline

almacenamiento optimizado para el algoritmo

Es la forma eficiente de pasar datos a CatBoost.

🧪 8. Evaluación del modelo

Métrica usada: MAE (Mean Absolute Error)
Interpretación:

error medio absoluto entre el precio real y el predicho.

En tus pruebas, el MAE fue lo suficientemente bajo como para considerar el modelo apto para predicciones de mercado.

🌡 9. Análisis de depreciación (slope €/año)

Se calcula ajustando una regresión precio ~ año para cada modelo.

Un slope alto (positivo) → coche que mantiene precio
Un slope negativo → coche que pierde valor rápido

Esto se integró en la app para ofrecer:

ranking por modelo

ranking por marca

modelos similares al tuyo

estimación de revalorización anual

📱 10. App interactiva – Streamlit

Incluye:

🔧 Pestaña Predicción

Inputs del usuario

Predicción en tiempo real del precio

Gráfica precio vs kilómetros

Gráfica precio vs año con proyección al 2025

Comparación con coches similares

💼 Pestaña Inversión

Explicación del concepto pendiente anual

Rankings dinámicos

Modelos que mejor mantienen valor

Interpretaciones de mercado

📈 11. Visualizaciones incluidas

Heatmap de correlación

Gráficas Plotly interactivas

Comparación de tu coche dentro del ranking

Líneas verticales marcando:

tus km

tu año

el precio predicho actual

🚀 12. Mejoras futuras
✔ Ampliación del dataset

más años (2021–2025)

más marcas premium

coches eléctricos

incluir variables económicas externas (IPC, inflación, interés)

✔ Más modelos ML avanzados

LightGBM

Optuna para hiperparámetros

Modelos híbridos: CatBoost + Redes

✔ Más funcionalidad en la app

estimación de precio futuro por km + año

comparador entre dos coches

simulador de compra-venta

alerta de buenas oportunidades en Wallapop/Milanuncios

📂 13. Estructura del repositorio
.
├── app/                           # Código de la aplicación Streamlit
│   └── app.py                     # Backend + interfaz de usuario
│
├── Data/                          # Datos originales por marca/modelo
│   ├── audi.csv
│   ├── bmw.csv
│   ├── cclass.csv
│   ├── focus.csv
│   ├── ford.csv
│   ├── hyundai.csv
│   ├── mercedes.csv
│   ├── Opel.csv
│   ├── skoda.csv
│   ├── toyota.csv
│   ├── volkswagen.csv
│   ├── unclean_cclass.csv         # Datos sin limpiar
│   ├── unclean_focus.csv
│   └── ...                        # Otros datasets por modelo
│
├── cars_clean.csv                 # Dataset final unificado y limpio
│
├── CatBoost.ipynb                 # Notebook con EDA + entrenamiento del modelo
│
├── catboost_cars.cbm              # Modelo CatBoost entrenado y guardado
│
├── catboost_info/                 # Logs automáticos generados por CatBoost
│
├── requirements.txt               # Dependencias necesarias para ejecutar el proyecto
│
└── README.md                      # (Pendiente) Documentación principal del proyecto


🔧 14. Instalación y ejecución
pip install -r requirements.txt
streamlit run app.py

🎯 15. Conclusiones

Has construido un pipeline completo:

limpieza

EDA

ingeniería de características

entrenamiento CatBoost

evaluación MAE

visualizaciones avanzadas

app final usable por usuarios reales

Es un proyecto muy sólido a nivel Data Science, perfectamente defendible en una presentación técnica o en un portfolio profesional.