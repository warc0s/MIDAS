# streamlit_app.py
import streamlit as st
import base64
import chardet
from flow import FlowPlotV1
import pandas as pd
import time
from io import BytesIO
import os
from dotenv import load_dotenv

# --------------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Midas Plot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# CONSTANTES Y VARIABLES DE SESIÓN
# --------------------------------------------------------------------------------
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/1534/1534959.png"

if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = ""

if "csv_content" not in st.session_state:
    st.session_state["csv_content"] = None

# Para almacenar la imagen generada y preservar el estado
if "generated_image" not in st.session_state:
    st.session_state["generated_image"] = None

# --------------------------------------------------------------------------------
# CSS PERSONALIZADO (Dark Blue & Gold más oscuro)
# --------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --primary: #0A0F22;
        --secondary: #DAA520; /* Un dorado más oscuro que #FFD700 */
        --background: #121212;
    }
    
    .main {
        background: var(--background);
        color: #FFFFFF;
    }
    
    .header {
        background: var(--primary);
        padding: 1.5rem;
        color: white;
        margin-bottom: 2rem;
        border-bottom: 2px solid var(--secondary);
    }
    
    .sidebar .sidebar-content {
        background: var(--primary);
        color: white;
        padding: 1rem;
    }
    
    .stButton>button {
        background: var(--secondary) !important;
        color: var(--primary) !important;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        box-shadow: 0 2px 8px rgba(218,165,32,0.3);
    }
    
    .stFileUploader>div>div>div>div {
        border: 2px dashed var(--secondary) !important;
        border-radius: 8px;
        background: rgba(218,165,32,0.05);
    }
    
    .data-preview {
        border: 1px solid rgba(218,165,32,0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: var(--primary);
    }
    
    footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background: var(--primary);
        color: white;
        padding: 0.8rem;
        text-align: center;
        font-size: 0.9em;
    }
    
    .metric-box {
        background: rgba(218,165,32,0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .encoding-info {
        font-size: 0.9em;
        color: var(--secondary);
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------------------------------
def detect_encoding(content: bytes) -> str:
    """Detecta la codificación del archivo."""
    result = chardet.detect(content)
    return result['encoding'] or 'utf-8'

def validate_csv(content: bytes) -> bool:
    """Valida que el contenido sea un CSV legible."""
    try:
        encoding = detect_encoding(content)
        content_str = content.decode(encoding, errors='replace')
        pd.read_csv(BytesIO(content_str.encode(encoding)))
        return True
    except Exception:
        return False

def show_data_preview(content: bytes):
    """Muestra vista previa interactiva del CSV."""
    encoding = detect_encoding(content)
    content_str = content.decode(encoding, errors='replace')

    with st.expander("🔍 Vista Previa del CSV", expanded=True):
        df = pd.read_csv(BytesIO(content_str.encode(encoding)))
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-box">📈 Filas<br><strong>{df.shape[0]}</strong></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="metric-box">📊 Columnas<br><strong>{df.shape[1]}</strong></div>',
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f'<div class="metric-box">💾 Tamaño<br><strong>{len(content)/1024:.2f} KB</strong></div>',
                unsafe_allow_html=True
            )

def show_upload_section():
    """Muestra la sección de carga de archivos en la barra lateral."""
    with st.sidebar.expander("📤 Cargar Datos", expanded=True):
        uploaded_file = st.file_uploader(
            "Arrastra tu archivo CSV aquí",
            type="csv",
            help="Formatos soportados: CSV (codificación UTF-8, Latin-1)"
        )

        if uploaded_file:
            content = uploaded_file.getvalue()
            encoding = detect_encoding(content)
            st.markdown(
                f'<div class="encoding-info">Detectado encoding: <strong>{encoding}</strong></div>',
                unsafe_allow_html=True
            )

    return uploaded_file

def show_prompt_section():
    """Muestra la sección de entrada de prompt en la barra lateral."""
    with st.sidebar.expander("✍️ Descripción de la Visualización", expanded=True):
        user_prompt = st.text_area(
            "Describe tu visualización:",
            value=st.session_state["user_prompt"],
            height=120,
            placeholder="Ej: Gráfico de líneas comparando ventas y gastos por mes",
            help="Sé específico: tipo de gráfico, ejes, colores, estilo"
        )
        st.session_state["user_prompt"] = user_prompt

        st.markdown("**Ejemplos de prompts:**")
        st.markdown("- Gráfico de barras verticales mostrando ventas por mes")
        st.markdown("- Pie chart con distribución de gastos por categoría")

def get_friendly_error_message(e: Exception) -> str:
    """
    Transforma ciertos errores técnicos a mensajes más comprensibles para el usuario.
    """
    error_msg = str(e)
    if "string argument should contain only ASCII characters" in error_msg:
        return ("Parece que hay caracteres especiales en los datos que no son compatibles. "
                "Asegúrate de que el CSV utiliza una codificación adecuada, como UTF-8.")
    # Puedes agregar más casos específicos según sea necesario.
    return "Ocurrió un error durante el proceso. Por favor, revisa tus datos o la configuración."

# --------------------------------------------------------------------------------
# LÓGICA PRINCIPAL
# --------------------------------------------------------------------------------
def main():
    # Encabezado
    st.markdown(
        f"""
        <div class="header">
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <img src="{LOGO_URL}" width="60">
                <div>
                    <h1 style="margin:0; color: var(--secondary);">Midas Plot</h1>
                    <p style="margin:0; opacity: 0.8; font-size: 1.1rem;">Visual Analytics Platform</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar: carga de datos y prompt
    uploaded_file = show_upload_section()
    show_prompt_section()

    # Contenido principal
    if uploaded_file:
        # Guardamos el CSV en session_state para no perderlo tras rerun
        st.session_state["csv_content"] = uploaded_file.getvalue()

        # Validamos y mostramos vista previa
        if not validate_csv(st.session_state["csv_content"]):
            st.error("Formato CSV inválido o codificación no soportada.")
            return
        else:
            show_data_preview(st.session_state["csv_content"])

        # Si ya se generó la imagen, se muestra y se permite descargar sin perder el estado
        if st.session_state["generated_image"]:
            try:
                image_data = base64.b64decode(st.session_state["generated_image"])
                st.subheader("Resultado del Análisis")
                st.image(image_data, use_column_width=True, caption="Visualización Generada por IA")
                st.download_button(
                    label="📥 Exportar como PNG",
                    data=image_data,
                    file_name="midas_plot.png",
                    mime="image/png"
                )
                with st.expander("Código utilizado", expanded=False):
                    flow = FlowPlotV1(api_input={})
                    generated_code = flow.get_generated_code()
                    if generated_code:
                        st.code(generated_code, language="python")
                    else:
                        st.warning("No se pudo recuperar el código generado")
            except Exception as e:
                st.error(get_friendly_error_message(e))
                with st.expander("Detalles técnicos", expanded=False):
                    st.exception(e)
        else:
            if st.button("🚀 Generar Visualización", use_container_width=True):
                try:
                    with st.spinner("Generando gráfica, por favor espera..."):
                        encoding = detect_encoding(st.session_state["csv_content"])
                        csv_content_str = st.session_state["csv_content"].decode(encoding, errors='replace')

                        flow = FlowPlotV1(api_input={
                            'prompt': st.session_state["user_prompt"],
                            'csv_content': csv_content_str
                        })
                        base64_image = flow.kickoff()
                        st.session_state["generated_image"] = base64_image

                    if base64_image:
                        image_data = base64.b64decode(base64_image)
                        st.subheader("Resultado del Análisis")
                        st.image(image_data, use_column_width=True, caption="Visualización Generada por IA")
                        st.download_button(
                            label="📥 Exportar como PNG",
                            data=image_data,
                            file_name="midas_plot.png",
                            mime="image/png"
                        )
                        with st.expander("Código utilizado", expanded=False):
                            generated_code = flow.get_generated_code()
                            if generated_code:
                                st.code(generated_code, language="python")
                            else:
                                st.warning("No se pudo recuperar el código generado")
                except Exception as e:
                    st.error(get_friendly_error_message(e))
                    with st.expander("Detalles técnicos", expanded=False):
                        st.exception(e)
    else:
        st.info(
            """
            **Bienvenido a Midas Plot**  
            1. Sube tu archivo CSV en el panel izquierdo.  
            2. (Opcional) Describe la visualización requerida.  
            3. Pulsa "Generar Visualización" para ver tu gráfico.  
            """
        )

    # Footer
    st.markdown(
        """
        <footer>
            <div style="opacity: 0.8;">
                2024 Midas Plot | Versión 2.1 | 
                <a href="#privacy" style="color: var(--secondary); text-decoration: none;">Privacidad</a> | 
                <a href="#terms" style="color: var(--secondary); text-decoration: none;">Términos</a>
            </div>
        </footer>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
