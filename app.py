import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis datos Saber 11",
    page_icon="components/Logo_Saber11.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
col1, col2 = st.columns([1, 4.5])
with col1:
    st.write("")
    st.write("")
    st.write("")
    st.image("components/Logo_Saber11.png", width=200)
with col2:
    st.title("Plataforma de Análisis de resultados de las pruebas Saber 11")
st.markdown("---")

# Función para cargar datos
DATASET_PATH = "data/Saber11_modificado.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    return df
# Cargar datos
df = load_data()


# Mostrar información del dataset
st.subheader("📈 Información del Dataset")

col1, col2 = st.columns(2)
with col1:
    st.metric("Filas", df.shape[0])
with col2:
    st.metric("Columnas", df.shape[1])

# Mostrar muestra de datos
with st.expander("🔍 Ver muestra de datos"):
    st.dataframe(df.head(10))

# Estadísticas descriptivas
with st.expander("📊 Estadísticas Descriptivas"):
    st.dataframe(df.describe())

# =================== VISUALIZACIONES ===================
st.subheader("📊 Análisis Descriptivo")

tab1, tab2, tab3 = st.tabs(["Desempeños poblacionales", "Matriz de correlación", "Variables Socioeconómicas"]) #SE REFIERE A LAS PESTAÑAS

with tab1:
    st.markdown("### Visualizaciones")
    
    options = [
        "Ingles",
        "Matematicas",
        "Ciencias sociales y ciudadanas",
        "Ciencias naturales",
        "Lectura critica",
        "Desempeño Global"
    ]

    selected_option = st.selectbox("Selecciona una visualización:", options)
    
    if selected_option == "Ingles":
        st.image("components/nivelDesempenoIngles.png", caption="Nivel de Desempeño en Ingles", use_container_width=True)
    elif selected_option == "Matematicas":
        st.image("components/nivelDesempenoMatematicas.png", caption="Nivel de Desempeño en Matematicas", use_container_width=True)
    elif selected_option == "Ciencias sociales y ciudadanas":
        st.image("components/nivelDesempenoCienciasSociales.png", caption="Nivel de Desempeño en Ciencias Sociales", use_container_width=True)
    elif selected_option == "Ciencias naturales":
        st.image("components/nivelDesempenoCienciasNaturales.png", caption="Nivel de Desempeño en Ciencias Naturales", use_container_width=True)
    elif selected_option == "Lectura critica":
        st.image("components/nivelDesempenoLecturaCritica.png", caption="Nivel de Desempeño en Lectura Critica", use_container_width=True)
    elif selected_option == "Desempeño Global":
        st.image("components/nivelDesempenoGlobal.png", caption="Nivel de Desempeño Global", use_container_width=True)

    
   

with tab2:
    st.markdown("### Matriz de Correlación")

    df_codificado = pd.DataFrame(df, columns=['COLE_AREA_UBICACION','COLE_BILINGUE','COLE_CARACTER','COLE_GENERO','COLE_JORNADA','COLE_NATURALEZA',
    'COLE_SEDE_PRINCIPAL','ESTU_GENERO','FAMI_CUARTOSHOGAR','FAMI_EDUCACIONMADRE','FAMI_EDUCACIONPADRE','FAMI_ESTRATOVIVIENDA',
    'FAMI_PERSONASHOGAR','FAMI_TIENEAUTOMOVIL','FAMI_TIENECOMPUTADOR','FAMI_TIENEINTERNET','FAMI_TIENELAVADORA','PUNT_GLOBAL',
    'Ingles_Nivel','Matematicas_Nivel','Sociales_Nivel','Ciencias_Nivel','Lectura_Nivel','CLASIFICACIÓN'])

    columnas_a_estandarizar = ['COLE_AREA_UBICACION','COLE_BILINGUE','COLE_CARACTER','COLE_GENERO','COLE_JORNADA','COLE_NATURALEZA',
    'COLE_SEDE_PRINCIPAL','ESTU_GENERO','FAMI_CUARTOSHOGAR','FAMI_EDUCACIONMADRE','FAMI_EDUCACIONPADRE','FAMI_ESTRATOVIVIENDA',
    'FAMI_PERSONASHOGAR','FAMI_TIENEAUTOMOVIL','FAMI_TIENECOMPUTADOR','FAMI_TIENEINTERNET','FAMI_TIENELAVADORA','PUNT_GLOBAL',
    'Ingles_Nivel','Matematicas_Nivel','Sociales_Nivel','Ciencias_Nivel','Lectura_Nivel','CLASIFICACIÓN']

    columnas_reemplazo1 = ['COLE_AREA_UBICACION','COLE_BILINGUE','COLE_CARACTER','COLE_GENERO','COLE_JORNADA','COLE_NATURALEZA',
    'COLE_SEDE_PRINCIPAL','ESTU_GENERO','FAMI_CUARTOSHOGAR','FAMI_EDUCACIONMADRE','FAMI_EDUCACIONPADRE','FAMI_ESTRATOVIVIENDA',
    'FAMI_PERSONASHOGAR','FAMI_TIENEAUTOMOVIL','FAMI_TIENECOMPUTADOR','FAMI_TIENEINTERNET','FAMI_TIENELAVADORA','PUNT_GLOBAL',
    'Ingles_Nivel','Matematicas_Nivel','Sociales_Nivel','Ciencias_Nivel','Lectura_Nivel','CLASIFICACIÓN']

    mapeo_codificacion = {
    'Si': 1,
    'No': 0,
    'URBANO': 1,
    'RURAL': 0,
    'Femenino': 0,
    'Masculino': 1,
    'Uno': 1,
    'Dos': 2,
    'Tres': 3,
    'Cuatro': 4,
    'Cinco': 5,
    'Seis o mas': 6,
    'No Aplica': 0,
    'A-': 0,
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B+': 4,
    '1 a 2': 1,
    '3 a 4': 2,
    '5 a 6': 3,
    '7 a 8': 4,
    '9 o más': 5,
    'OFICIAL': 1,
    'NO OFICIAL': 0,
    'NO APLICA': 0,
    'ACADÉMICO': 1,
    'TÉCNICO/ACADÉMICO': 2,
    'TÉCNICO':  3,
    'MIXTO': 1,
    'MASCULINO': 2,
    'FEMENINO': 3,
    'No Aplica': 0,
    'No sabe': 0,
    'Ninguno': 0,
    'Primaria incompleta': 1,
    'Primaria completa': 2,
    'Secundaria (Bachillerato) incompleta': 3,
    'Secundaria (Bachillerato) completa': 4,
    'Técnica o tecnológica incompleta': 5,
    'Técnica o tecnológica completa': 6,
    'Educación profesional incompleta': 7,
    'Educación profesional completa': 8,
    'Postgrado': 9,
    'SABATINA': 0,
    'NOCHE': 1,
    'MAÑANA': 2,
    'TARDE': 3,
    'UNICA': 4,
    'COMPLETA': 5,
    'MENOR AL PUNT NACIONAL':0,
    'SUPERIOR AL PUNT NACIONAL':1,
    'BECA GENERACIÓN E':2,
}
    for columna in columnas_reemplazo1:
        df_codificado[columna].replace(mapeo_codificacion, inplace=True)
    
    df_codificado = df_codificado.astype(float)

    scaler = StandardScaler()
    df_estandarizado_array = scaler.fit_transform(df_codificado[columnas_a_estandarizar])

    df_estandarizado = pd.DataFrame(df_estandarizado_array, columns=columnas_a_estandarizar)
    
    # Calcular correlación
    matriz_correlacion = df_estandarizado.corr()

    fig_corr = px.imshow(
        matriz_correlacion,
        text_auto='.4f',
        aspect="auto",
        color_continuous_scale="RdBu_r",
        
    )

    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.markdown("### Visualizaciones")
    
    options2 = [
        "Porcentaje de estudiantes con internet",
        "Porcentaje de estudiantes con lavadora",
        "Porcentaje de estudiantes por nivel de educación de la madre",
        "Porcentaje de estudiantes por nivel de educación del padre",
        "Promedio de puntaje global por estrato socioeconomico"
    ]

    selected_option2 = st.selectbox("Selecciona una visualización:", options2)

    if selected_option2 == "Porcentaje de estudiantes con internet":
        st.image("components/internetYear.png", caption="% de estudiantes con y sin internet", use_container_width=True)
    elif selected_option2 == "Porcentaje de estudiantes con lavadora":
        st.image("components/lavadoraYear.png", caption="% de estudiantes con y sin lavadora", use_container_width=True)
    elif selected_option2 == "Porcentaje de estudiantes por nivel de educación de la madre":
        st.image("components/educacionMadre.png", caption="% de estudiantes por nivel de educación de la madre", use_container_width=True)
    elif selected_option2 == "Porcentaje de estudiantes por nivel de educación del padre":
        st.image("components/educacionPadre.png", caption="% de estudiantes por nivel de educación del padre", use_container_width=True)
    elif selected_option2 == "Promedio de puntaje global por estrato socioeconomico":
        st.image("components/puntajeEstrato.png", caption="Promedio de puntaje global por estrato socioeconomico", use_container_width=True)

# =================== MACHINE LEARNING ===================
st.markdown("---")
st.subheader("🤖 Análisis Predictivo")


# Tamaño del conjunto de prueba
# test_size = st.sidebar.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.2, 0.05) #Orden de los parametros: Min, Max, Predeterminado, paso


def categorizar_puntaje(puntaje):
    if puntaje < 349:
        return "No Becado"
    else:
        return "Becado"
    
df_codificado["RENDIMIENTO"] = df_codificado["PUNT_GLOBAL"].apply(categorizar_puntaje)

columnas_features = ["COLE_JORNADA",
                     "FAMI_EDUCACIONMADRE",
                     "FAMI_EDUCACIONPADRE",
                     "FAMI_ESTRATOVIVIENDA"]

X = df_codificado[columnas_features]
y = df_codificado["RENDIMIENTO"]

# ========================
# 4. División entrenamiento/prueba
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# 5. Entrenar modelo
# ========================
modelo = RandomForestClassifier(random_state=42, max_depth=4, n_estimators=100)
modelo.fit(X_train, y_train)

# ========================
# 6. Predicciones
# ========================
y_pred = modelo.predict(X_test)

st.subheader("📊 Entrenamiento y testeo del modelo Random Forest")
col1, col2 = st.columns(2)

# Matriz de confusión en la columna 1
with col1:
    st.markdown("#### 🔎 Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6,5.95))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=modelo.classes_,
                yticklabels=modelo.classes_,
                ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión RF")
    st.pyplot(fig)

# Reporte de clasificación en la columna 2
with col2:
    st.markdown("#### 📑 Reporte de Clasificación")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))

    st.subheader("🌟 Importancia de las Variables")

    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    importancias = importancias.sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.barplot(x=importancias, y=importancias.index, palette="viridis", ax=ax2)
    ax2.set_title("Importancia de las Variables")
    ax2.set_xlabel("Importancia")
    ax2.set_ylabel("Variables")
    st.pyplot(fig2)

    
# Predicción interactiva
st.markdown("---")
st.markdown("### 🎯 Haz tu propia predicción")

st.markdown("Ajusta los valores para hacer una predicción:")


col1, col2 = st.columns(2)

with col1:
    cole_jornada = st.selectbox(
        "Jornada del Colegio",
        options=sorted(df["COLE_JORNADA"].dropna().unique())
    )

    fami_edu_padre = st.selectbox(
        "Educación del padre",
        options=sorted(df["FAMI_EDUCACIONPADRE"].dropna().unique())
    )

with col2:
    fami_estrato = st.selectbox(
        "Estrato de la vivienda",
        options=sorted(df["FAMI_ESTRATOVIVIENDA"].dropna().unique())
    )

    fami_edu_madre = st.selectbox(
        "Educación de la madre",
        options=sorted(df["FAMI_EDUCACIONMADRE"].dropna().unique())
    )

# Botón de predicción
if st.button("Predecir"):
    # Crear DataFrame con los valores seleccionados
    input_data = pd.DataFrame({
        "COLE_JORNADA": [cole_jornada],
        "FAMI_EDUCACIONMADRE": [fami_edu_madre],
        "FAMI_EDUCACIONPADRE": [fami_edu_padre],
        "FAMI_ESTRATOVIVIENDA": [fami_estrato]
    })


    input_data = input_data.replace(mapeo_codificacion).astype(float)
    # Hacer predicción
    prediction = modelo.predict(input_data)[0]
    proba = modelo.predict_proba(input_data)[0]

    st.success(f"🎯 Predicción: **{prediction}**")

    # Mostrar probabilidades
    st.markdown("**Probabilidades:**")
    prob_df = pd.DataFrame({
        "Clase": modelo.classes_,
        "Probabilidad": proba
    })
    st.bar_chart(prob_df.set_index("Clase"))


# Footer
st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        <h3>⚠️Importante⚠️</h3>
        Esta herramienta ofrece estimaciones sobre la posibilidad de obtener una beca a partir de predicciones de resultados Saber 11. El modelo fue entrenado con datos históricos y utiliza variables socioeconómicas únicamente como insumos estadísticos, las cuales no determinan tu resultado final. La actitud y práctica de aprendizaje del estudiante son factores decisivos para su desempeño real. No garantiza la concesión de becas ni reemplaza procesos oficiales del ICFES o de las instituciones. Usa los resultados como orientación, no como decisión definitiva
    </div>
    """, 
    unsafe_allow_html=True
)