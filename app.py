import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# --- Configuración de la Aplicación ---
st.set_page_config(page_title="Predicción de Churn de Clientes Telco", layout="wide")

# Rutas para guardar/cargar los objetos
PREPROCESSOR_PATH = 'preprocessor.joblib'
LOGREG_MODEL_PATH = 'logreg_model.joblib'
KNN_MODEL_PATH = 'knn_model.joblib'

# Definición de tipos de columnas para preprocesamiento
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# --- Funciones de Entrenamiento y Preprocesamiento (Ejecutar solo si es necesario) ---

@st.cache_resource
def train_and_save_models():
    """
    Simula la carga, preprocesamiento y entrenamiento de modelos. 
    Esto se ejecuta una sola vez para guardar los objetos necesarios.
    """
    st.info("Entrenando modelos y guardando preprocesador (solo la primera vez)...")
    try:
        # Asegúrate de que este archivo CSV esté en el mismo directorio
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv') 
    except FileNotFoundError:
        st.error("Archivo 'WA_Fn-UseC_-Telco-Customer-Churn.csv' no encontrado. Asegúrate de que esté en el mismo directorio.")
        return None, None, None

    # 1. Limpieza y preparación de datos
    df = df.drop(columns=['customerID'])
    # Manejar 'TotalCharges' (espacios vacíos -> NaN -> Drop)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # 2. Variable Target
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn']) # No=0, Yes=1
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Dividir datos (necesario para el entrenamiento)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Creación del Preprocesador (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             [col for col in X_train.columns if col not in NUMERICAL_FEATURES])
        ],
        remainder='passthrough'
    )
    
    # 4. Creación de Pipelines y Entrenamiento

    # Pipeline de Regresión Logística
    logreg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(solver='liblinear', random_state=42))])
    logreg_pipe.fit(X_train, y_train)

    # Pipeline de KNN
    # Usando n_neighbors=21 y p=1 (Manhattan distance) como ejemplo de un modelo entrenado
    knn_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier(n_neighbors=21, p=1))]) 
    knn_pipe.fit(X_train, y_train)

    # 5. Guardar modelos y preprocesador
    joblib.dump(logreg_pipe, LOGREG_MODEL_PATH)
    joblib.dump(knn_pipe, KNN_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    st.success("Modelos y preprocesador guardados correctamente. ¡Listo para usar!")
    return logreg_pipe, knn_pipe, preprocessor

# --- Función para Cargar Modelos ---
@st.cache_resource
def load_models():
    """
    Carga los modelos entrenados y el preprocesador.
    """
    try:
        logreg_model = joblib.load(LOGREG_MODEL_PATH)
        knn_model = joblib.load(KNN_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return logreg_model, knn_model, preprocessor
    except FileNotFoundError:
        # Si no existen, se entrenan y guardan por primera vez
        return train_and_save_models()

logreg_model, knn_model, preprocessor = load_models()

if logreg_model is None:
    st.stop()


# --- Interfaz de Usuario (Formulario de Entrada) ---
st.title("Aplicación Web de Predicción de Churn de Clientes Telco")
st.markdown("---")

st.header("Formulario de Variables del Cliente")

with st.form("churn_prediction_form"):
    
    # Diseño en Columnas
    col1, col2, col3 = st.columns(3)

    # Columna 1: Información Demográfica y de Contrato
    with col1:
        st.subheader("1. Demografía y Contrato")
        
        gender = st.selectbox("Género", ('Female', 'Male'))
        senior_citizen = st.selectbox("Ciudadano Mayor (SeniorCitizen)", (0, 1), format_func=lambda x: 'Sí' if x==1 else 'No')
        partner = st.selectbox("Pareja (Partner)", ('Yes', 'No'))
        dependents = st.selectbox("Dependientes", ('Yes', 'No'))
        
        tenure = st.slider("Antigüedad (Tenure) en meses", 1, 72, 12)
        contract = st.selectbox("Tipo de Contrato", ('Month-to-month', 'One year', 'Two year'))
        paperless_billing = st.selectbox("Facturación sin Papel (PaperlessBilling)", ('Yes', 'No'))
        payment_method = st.selectbox("Método de Pago", 
                                      ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))


    # Columna 2: Servicios de Teléfono e Internet
    with col2:
        st.subheader("2. Servicios Principales y Adicionales")
        
        phone_service = st.selectbox("Servicio Telefónico", ('Yes', 'No'))
        
        if phone_service == 'Yes':
            multiple_lines = st.selectbox("Líneas Múltiples", ('Yes', 'No'))
        else:
            multiple_lines = 'No phone service'
            st.markdown("*(Líneas Múltiples: No phone service)*")
        
        internet_service = st.selectbox("Servicio de Internet", ('DSL', 'Fiber optic', 'No'))
        
        if internet_service != 'No':
            online_security = st.selectbox("Seguridad Online", ('Yes', 'No'))
            online_backup = st.selectbox("Copia de Seguridad Online", ('Yes', 'No'))
            device_protection = st.selectbox("Protección de Dispositivo", ('Yes', 'No'))
            tech_support = st.selectbox("Soporte Técnico", ('Yes', 'No'))
            streaming_tv = st.selectbox("TV por Streaming", ('Yes', 'No'))
            streaming_movies = st.selectbox("Películas por Streaming", ('Yes', 'No'))
            
        else:
            # Si no hay InternetService, los adicionales son "No internet service"
            online_security = 'No internet service'
            online_backup = 'No internet service'
            device_protection = 'No internet service'
            tech_support = 'No internet service'
            streaming_tv = 'No internet service'
            streaming_movies = 'No internet service'
            st.markdown("*(Servicios Adicionales: No internet service)*")

    # Columna 3: Información de Cargos
    with col3:
        st.subheader("3. Cargos")
        
        monthly_charges = st.number_input("Cargo Mensual (MonthlyCharges)", min_value=18.25, max_value=118.75, value=50.00, step=0.01)
        total_charges = st.number_input("Cargo Total (TotalCharges)", min_value=18.80, max_value=8684.80, value=1000.00, step=0.01)
        
    # Botón de Predicción
    submitted = st.form_submit_button("Realizar Predicción")


if submitted:
    
    # 1. Crear DataFrame con las entradas del usuario
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    st.markdown("---")
    st.header("Resultados de la Predicción")
    colA, colB = st.columns(2)

    # --- 1. Probar el modelo de Regresión Logística ---
    with colA:
        st.subheader("1. Modelo de Regresión Logística")
        
        # Predicción de Probabilidad
        churn_proba = logreg_model.predict_proba(input_data)[:, 1][0]
        
        # Clasificación
        churn_pred = logreg_model.predict(input_data)[0]
        
        st.markdown(f"**Probabilidad de Churn:** **`{churn_proba:.2%}`**")
        
        if churn_pred == 1:
            st.error(f"**Clasificación (Churn):** **`Yes`**")
        else:
            st.success(f"**Clasificación (Churn):** **`No`**")
            
    # --- 2. Probar el modelo KNN ---
    with colB:
        st.subheader("2. Modelo K-Nearest Neighbors (KNN)")
        
        # Clasificación KNN
        knn_pred = knn_model.predict(input_data)[0]
        
        st.markdown(f"**Predicción del Vecino (Clasificación):**")
        
        if knn_pred == 1:
            st.error(f"**Clasificación (Churn):** **`Yes`**")
        else:
            st.success(f"**Clasificación (Churn):** **`No`**")

    st.markdown("---")
    st.caption("Nota: Los modelos se entrenan automáticamente si no existen los archivos `joblib`.")