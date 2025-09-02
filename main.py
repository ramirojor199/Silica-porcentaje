
import joblib
import streamlit as st
import pandas as pd

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Predictor de porcentaje de Silica",
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
model = load_model('model.joblib')

# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos del proceso de lixivacion.
    """)

    # Slider para el caudal de alimentación
    ironconcentrate = st.slider(
        label='% Iron Concentrate)',
        min_value=0,
        max_value=100,
        value=50, # Valor inicial
        step=1
    )
    st.caption("Representa el porcentaje de Hierro que se alimenta al proceso.")

    # Slider para el flujo de Amina
    Amina = st.slider(
        label='Amina Flow ()',
        min_value=100,
        max_value=800,
        value=400,
        step=1
    )
    st.caption("El flujo de Amina es escencial para la extracción de silicatos.")

    # Slider para la columna de flujo de aire 01
    airflow = st.slider(
        label='Flotation Column 01 Air',
        min_value=100,
        max_value=500,
        value=200,
        step=1
    )
    st.caption("El flujo de aire representa una carga signitificativa en el proceso de lixiviacion.")

# --- Contenido de la Página Principal ---
st.title("🧪 Predictor de Porcentaje de Silica")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza un modelo de machine learning para predecir el rendimiento de un producto químico en un proceso de lixiviacion en parámetros operativos clave.

**Esta herramienta puede ayudar a los ingenieros de procesos y operadores a:**
- **Optimizar** las condiciones de operación para obtener el máximo rendimiento.
- **Predecir** el impacto de los cambios en el proceso antes de implementarlos.
- **Solucionar** problemas potenciales simulando diferentes escenarios.
""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir Rendimiento', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        df_input = pd.DataFrame({
            '% Iron Concentrate': [ironconcentrate],
            'Amina Flow': [Amina],
            'Flotation Column 01 Air': [airflow]
        })

        # Hacemos la predicción
        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"**Rendimiento Predicho:** `{prediction_value[0]:.2f}%`")
            st.info("Este valor representa el porcentaje estimado del producto deseado que se recuperará.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  **Predicción:** El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  **Resultado:** La aplicación muestra el rendimiento final predicho como un porcentaje.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `Regression Model` (XGBoost Optimizado)
    * **Propósito:** Predecir el valor continuo del rendimiento del porcentaje de Silica.
    * **Características Usadas:** Porcentaje de Hierro, Flujo de Amina y Columna de flotacion de flujo de aire 01 .
    """)
