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
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🤖 Dashboard de Machine Learning")
st.markdown("---")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Selección de dataset
dataset_option = st.sidebar.selectbox(
    "📊 Selecciona el Dataset:",
    ["Iris", "Wine", "Dataset Sintético"]
)

# Función para cargar datos
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df, data.target_names
    
    elif dataset_name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['wine_class'] = df['target'].map({0: 'Class_0', 1: 'Class_1', 2: 'Class_2'})
        return df, data.target_names
    
    else:  # Dataset Sintético
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                                 n_informative=8, n_redundant=2, random_state=42)
        feature_names = [f'feature_{i+1}' for i in range(10)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['class'] = df['target'].map({0: 'Class_A', 1: 'Class_B', 2: 'Class_C'})
        return df, ['Class_A', 'Class_B', 'Class_C']

# Cargar datos
df, target_names = load_data(dataset_option)

# Mostrar información del dataset
st.subheader("📈 Información del Dataset")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filas", df.shape[0])
with col2:
    st.metric("Columnas", df.shape[1])
with col3:
    st.metric("Features", df.shape[1] - 2)  # -2 por target y clase
with col4:
    st.metric("Clases", len(target_names))

# Mostrar muestra de datos
with st.expander("🔍 Ver muestra de datos"):
    st.dataframe(df.head(10))

# Estadísticas descriptivas
with st.expander("📊 Estadísticas Descriptivas"):
    st.dataframe(df.describe())

# =================== VISUALIZACIONES ===================
st.subheader("📊 Visualizaciones")

tab1, tab2, tab3, tab4 = st.tabs(["Distribuciones", "Correlaciones", "Scatter Plots", "PCA"]) #SE REFIERE A LAS PESTAÑAS

with tab1:
    st.markdown("### Distribución de Variables")
    
    # Seleccionar variable para histograma
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'target']
    
    selected_column = st.selectbox("Selecciona una variable:", numeric_columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma con Plotly
        fig_hist = px.histogram(df, x=selected_column, color=df.columns[-1], 
                               title=f'Distribución de {selected_column}',
                               marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Boxplot con Plotly
        fig_box = px.box(df, y=selected_column, color=df.columns[-1],
                        title=f'Boxplot de {selected_column}')
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.markdown("### Matriz de Correlación")
    
    # Calcular correlación
    corr_matrix = df[numeric_columns].corr()
    
    # Plotly heatmap
    fig_corr = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        title="Matriz de Correlación",
                        color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.markdown("### Scatter Plots")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Eje X:", numeric_columns, key="scatter_x")
    with col2:
        y_axis = st.selectbox("Eje Y:", numeric_columns, key="scatter_y", 
                             index=1 if len(numeric_columns) > 1 else 0)
    
    # Scatter plot con Plotly
    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, 
                           color=df.columns[-1],
                           title=f'{x_axis} vs {y_axis}',
                           hover_data=numeric_columns[:3])
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    st.markdown("### Análisis de Componentes Principales (PCA)")
    
    # Preparar datos para PCA
    X = df[numeric_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Crear DataFrame con componentes principales
    pca_df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
    pca_df['target'] = df[df.columns[-1]].values
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Varianza explicada
        explained_variance = pca.explained_variance_ratio_
        fig_var = px.bar(x=range(1, len(explained_variance) + 1), 
                        y=explained_variance,
                        title="Varianza Explicada por Componente",
                        labels={'x': 'Componente Principal', 'y': 'Varianza Explicada'})
        st.plotly_chart(fig_var, use_container_width=True)
    
    with col2:
        # PCA 2D
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='target',
                           title="PCA - Primeros 2 Componentes")
        st.plotly_chart(fig_pca, use_container_width=True)

# =================== MACHINE LEARNING ===================
st.markdown("---")
st.subheader("🤖 Machine Learning")

# Configuración del modelo en sidebar
st.sidebar.markdown("---")
st.sidebar.header("🎯 Configuración ML")

# Selección de modelo
model_option = st.sidebar.selectbox(
    "Selecciona el Modelo:",
    ["Random Forest", "Logistic Regression", "SVM"]
)

# Tamaño del conjunto de prueba
test_size = st.sidebar.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.2, 0.05) #Orden de los parametros: Min, Max, Predeterminado, paso

# Preparar datos para ML
X = df[numeric_columns].values
y = df['target'].values

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                   random_state=42, stratify=y)

# Escalado de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configuración específica del modelo
if model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Número de árboles:", 10, 200, 100, 10)
    max_depth = st.sidebar.slider("Profundidad máxima:", 1, 20, 10, 1)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                 random_state=42)
    
elif model_option == "Logistic Regression":
    C = st.sidebar.slider("Parámetro C:", 0.01, 10.0, 1.0, 0.01)
    model = LogisticRegression(C=C, random_state=42, max_iter=1000)
    
else:  # SVM
    C = st.sidebar.slider("Parámetro C:", 0.01, 10.0, 1.0, 0.01)
    kernel = st.sidebar.selectbox("Kernel:", ["rbf", "linear", "poly"])
    model = SVC(C=C, kernel=kernel, random_state=42)

# Botón para entrenar modelo
if st.sidebar.button("🚀 Entrenar Modelo", type="primary"):
    with st.spinner("Entrenando modelo..."):
        # Entrenar modelo
        if model_option in ["Logistic Regression", "SVM"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        # Guardar resultados en session state
        st.session_state.model_trained = True
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.accuracy = accuracy
        st.session_state.y_pred = y_pred
        st.session_state.y_test = y_test
        st.session_state.target_names = target_names
        st.session_state.numeric_columns = numeric_columns

# Mostrar resultados si el modelo está entrenado
if hasattr(st.session_state, 'model_trained') and st.session_state.model_trained:
    
    # Métricas principales
    st.markdown("### 📊 Resultados del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{st.session_state.accuracy:.3f}")
    with col2:
        st.metric("Datos de Entrenamiento", len(X_train))
    with col3:
        st.metric("Datos de Prueba", len(X_test))
    with col4:
        st.metric("Features", len(numeric_columns))
    
    # Matriz de confusión
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Matriz de Confusión")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        
        fig_cm = px.imshow(cm, 
                          text_auto=True,
                          aspect="auto",
                          title="Matriz de Confusión",
                          labels=dict(x="Predicción", y="Real"),
                          x=st.session_state.target_names,
                          y=st.session_state.target_names)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("#### Reporte de Clasificación")
        report = classification_report(st.session_state.y_test, st.session_state.y_pred, 
                                     target_names=st.session_state.target_names,
                                     output_dict=True)
        
        # Convertir reporte a DataFrame
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))
    
    # Feature importance (solo para Random Forest)
    if model_option == "Random Forest":
        st.markdown("#### 🌟 Importancia de las Variables")
        
        feature_importance = st.session_state.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': st.session_state.numeric_columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h',
                        title="Importancia de las Variables")
        st.plotly_chart(fig_imp, use_container_width=True)
    
    # Predicción interactiva
    st.markdown("---")
    st.markdown("### 🎯 Hacer Predicción")
    
    st.markdown("Ajusta los valores para hacer una predicción:")
    
    # Crear inputs para cada feature
    prediction_inputs = {}
    cols = st.columns(min(3, len(st.session_state.numeric_columns)))
    
    for i, feature in enumerate(st.session_state.numeric_columns):
        col_idx = i % len(cols)
        with cols[col_idx]:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            
            prediction_inputs[feature] = st.slider(
                f"{feature}:",
                min_val, max_val, mean_val,
                key=f"pred_{feature}"
            )
    
    if st.button("🔮 Predecir", type="primary"):
        # Preparar datos para predicción
        input_data = np.array([[prediction_inputs[feature] for feature in st.session_state.numeric_columns]])
        
        if model_option in ["Logistic Regression", "SVM"]:
            input_data = st.session_state.scaler.transform(input_data)
        
        # Hacer predicción
        prediction = st.session_state.model.predict(input_data)[0]
        
        if hasattr(st.session_state.model, 'predict_proba'):
            if model_option in ["Logistic Regression", "SVM"]:
                probabilities = st.session_state.model.predict_proba(input_data)[0]
            else:
                probabilities = st.session_state.model.predict_proba(input_data)[0]
        else:
            probabilities = None
        
        # Mostrar resultado
        st.success(f"**Predicción: {st.session_state.target_names[prediction]}**")
        
        if probabilities is not None:
            st.markdown("**Probabilidades:**")
            prob_df = pd.DataFrame({
                'Clase': st.session_state.target_names,
                'Probabilidad': probabilities
            })
            
            fig_prob = px.bar(prob_df, x='Clase', y='Probabilidad',
                             title="Probabilidades de Predicción")
            st.plotly_chart(fig_prob, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        Desarrollado con ❤️ usando Streamlit, Pandas, Scikit-learn y Plotly
    </div>
    """, 
    unsafe_allow_html=True
)