
import joblib
import streamlit as st
import pandas as pd

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Modelo Predictivo del Porcentaje de Silica",
    page_icon="🧪",
    layout="wide"
)

# --- Carga del Modelo ---
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y se mantenga en memoria,
# lo que hace que la aplicación sea mucho más rápida.
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado desde un archivo .joblib."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {model_path}. Asegúrate de que el archivo del modelo esté en el directorio correcto.")
        return None

# Cargamos nuestro modelo campeón. Streamlit buscará en la ruta 'model.joblib'.
# NOTE: The model file name should be 'final_model.joblib' as mentioned in the markdown cell.
model = load_model('model.joblib')

# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos de la lixiviación.
    """)

    # Slider para el flujo de Amina
    AminaFlow = st.slider(
        label='Flujo de amina',
        min_value=240,
        max_value=750,
        value=400,#Valor Inicial
        step=1
    )
    st.caption("Representa el flujo de amina que afecta la concentracion de silica")

    # Slider para el % iron concentrate
ironconcentrate = st.slider(
        label='Porcentaje de concentración de hierro',
        min_value=60.00,
        max_value=70.00,
        step=0.005,
        format="%.2f"
    )
    st.caption("Porcentaje de hierrro que se alimenta al proceso.")

    # Slider para la Flotation Column 01 Air Flow
    flotationcolumnairflow = st.slider(
        label='Flujo de aire en la columa de flotación',
        min_value=-100,
        max_value=400,
        value=250,
        step=1
    )
    st.caption("Flujo de aire en la columna de flotación 01")

# --- Contenido de la Página Principal ---
st.title("🧪 Modelo Predictivo del Porcentaje de Silica")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza un modelo de machine learning para predecir el porcentaje de concentración de silica en el proceso de lixiviación basándose en parámetros operativos clave.

Esta herramienta puede ayudar a los ingenieros de procesos y operadores a:
- Optimizar las condiciones de operación para obtener el porcentage de silica final.
- Predecir el impacto de los cambios en el proceso antes de implementarlos.
- Solucionar problemas potenciales simulando diferentes escenarios.
""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir el porcentaje de silica', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        # The order of columns must match the order the model was trained on.
        # Based on the error message, the expected order is:
        # ['Amina Flow', 'Flotation Column 01 Air Flow', '% Iron Concentrate']
        df_input = pd.DataFrame({
            'Amina Flow': [AminaFlow],
            'Flotation Column 01 Air Flow': [flotationcolumnairflow],
            '% Iron Concentrate': [ironconcentrate]
        })

        # Hacemos la predicción
        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"Porcentaje Predicho: {prediction_value[0]:.2f}%")
            st.info("Este valor representa el porcentaje de silica presente en el operación.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ Sobre la Aplicación"):
    st.markdown("""
    ¿Cómo funciona?

    1.  Datos de Entrada: Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  Predicción: El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  Resultado: La aplicación muestra el porcentaje final predicho.

    Detalles del Modelo:

    * Tipo de Modelo: Regression Model (XGBoost Optimizado)
    * Propósito: Predecir el valor continuo del rendimiento de la destilación.
    * Características Usadas: Porcentaje de concentración de acero, Flujo de amina y Flujo de aire en la columna de flotación 01.
    """)
