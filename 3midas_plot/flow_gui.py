# streamlit_app.py
import streamlit as st
import base64
import chardet
from flow import FlowPlotV1
import pandas as pd
import time
from io import BytesIO, StringIO
import os
from dotenv import load_dotenv
import json
import traceback
import sys

# --------------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Midas Plot | Midas System",
    page_icon="https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/plot_trans.png?raw=true",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# CONSTANTES Y VARIABLES DE SESIÓN
# --------------------------------------------------------------------------------
LOGO_URL = "https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/plot_trans.png?raw=true"

if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = ""

if "csv_content" not in st.session_state:
    st.session_state["csv_content"] = None

# Para almacenar la imagen generada y preservar el estado
if "generated_image" not in st.session_state:
    st.session_state["generated_image"] = None

# Para almacenar el código generado
if "generated_code" not in st.session_state:
    st.session_state["generated_code"] = None

# Para almacenar el dataframe procesado
if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = None

# --------------------------------------------------------------------------------
# CSS PERSONALIZADO
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
    
    .action-buttons {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .reset-button button {
        background: #333 !important;
        color: var(--secondary) !important;
    }
    
    .reset-button button:hover {
        background: #444 !important;
    }
    
    .error-log {
        font-family: monospace;
        background: rgba(255, 0, 0, 0.1);
        padding: 10px;
        border-radius: 4px;
        color: #ff6b6b;
        white-space: pre-wrap;
        font-size: 0.85em;
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

def process_csv(content: bytes):
    """
    Procesa el CSV con manejo robusto de codificaciones.
    Retorna el dataframe y la codificación detectada.
    """
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    # Primero intentamos con la detección automática
    detected_encoding = detect_encoding(content)
    encodings_to_try = [detected_encoding] + [e for e in encodings_to_try if e != detected_encoding]
    
    last_error = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(BytesIO(content), encoding=encoding)
            # Usar nombres de columnas normalizados para evitar problemas
            df.columns = df.columns.str.replace('[^\w\s]', '', regex=True).str.strip()
            return df, encoding
        except Exception as e:
            last_error = e
            continue
    
    # Si llegamos aquí, ninguna codificación funcionó
    raise ValueError(f"No se pudo leer el CSV con ninguna codificación: {str(last_error)}")

def validate_csv(content: bytes) -> bool:
    """Valida que el contenido sea un CSV legible."""
    try:
        df, _ = process_csv(content)
        st.session_state["dataframe"] = df
        return True
    except Exception:
        return False

def show_data_preview(df, encoding, content_size):
    """Muestra vista previa interactiva del CSV."""
    with st.expander("🔍 Vista Previa del CSV", expanded=True):
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
                f'<div class="metric-box">💾 Tamaño<br><strong>{content_size/1024:.2f} KB</strong></div>',
                unsafe_allow_html=True
            )
        
        st.markdown(
            f'<div class="encoding-info">Codificación: <strong>{encoding}</strong></div>',
            unsafe_allow_html=True
        )

def show_upload_section():
    """Muestra la sección de carga de archivos en la barra lateral."""
    with st.sidebar.expander("📤 Cargar Datos", expanded=True):
        uploaded_file = st.file_uploader(
            "Arrastra tu archivo CSV aquí",
            type="csv",
            help="Formatos soportados: CSV (codificación UTF-8, Latin-1, ISO-8859-1, CP1252)"
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

def dataframe_to_csv(df):
    """
    Convierte un dataframe a CSV en formato de bytes sin caracteres problemáticos.
    """
    # Convertir a CSV con nombres de columnas simplificados
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def reset_visualization():
    """Resetea solo la visualización, manteniendo el dataset y prompt."""
    if "generated_image" in st.session_state:
        st.session_state["generated_image"] = None
    if "generated_code" in st.session_state:
        st.session_state["generated_code"] = None
    st.rerun()

def reset_everything():
    """Resetea todo: dataset, prompt, visualización y código."""
    for key in ["csv_content", "user_prompt", "generated_image", "generated_code", "dataframe"]:
        if key in st.session_state:
            st.session_state[key] = None
    st.rerun()

def generate_visualization(df, prompt):
    """
    Función para generar la visualización con manejo robusto de errores.
    """
    try:
        # Convertir el dataframe a string CSV
        csv_content = dataframe_to_csv(df)
        
        # Imprimir información de diagnóstico
        st.session_state["debug_info"] = {
            "columns": list(df.columns),
            "rows": len(df),
            "prompt_length": len(prompt),
            "csv_length": len(csv_content)
        }
        
        # Crear la instancia de FlowPlotV1 con los datos
        flow = FlowPlotV1(api_input={
            'prompt': prompt,
            'csv_content': csv_content
        })
        
        # Ejecutar kickoff para generar la imagen
        base64_image = flow.kickoff()
        
        # Si llegamos aquí, fue exitoso
        st.session_state["generated_image"] = base64_image
        
        # Obtener el código generado
        try:
            generated_code = flow.get_generated_code()
            if generated_code:
                st.session_state["generated_code"] = generated_code
        except Exception as e:
            st.warning(f"Se generó la visualización, pero no se pudo recuperar el código: {str(e)}")
        
        return True, None
    except Exception as e:
        # Capturar la traza completa para diagnóstico
        error_trace = traceback.format_exc()
        return False, {
            "error": str(e),
            "trace": error_trace,
            "type": str(type(e))
        }

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
    
    # Botón para resetear todo en la barra lateral (si hay datos cargados)
    if st.session_state["csv_content"] is not None:
        with st.sidebar:
            st.markdown("---")
            if st.button("🔄️ Pedir Nueva Grafica", use_container_width=True, help="Borra el dataset y todos los resultados"):
                reset_everything()

    # Contenido principal
    if uploaded_file or st.session_state["csv_content"] is not None:
        # Si se acaba de cargar un archivo, guardamos su contenido
        if uploaded_file:
            file_content = uploaded_file.getvalue()
            st.session_state["csv_content"] = file_content
            
            # Resetear la visualización al cargar un nuevo archivo
            if "generated_image" in st.session_state:
                st.session_state["generated_image"] = None
            if "generated_code" in st.session_state:
                st.session_state["generated_code"] = None
            
            # Procesar el CSV inmediatamente
            try:
                df, encoding = process_csv(file_content)
                st.session_state["dataframe"] = df
                show_data_preview(df, encoding, len(file_content))
            except Exception as e:
                st.error(f"Error al procesar el CSV: {str(e)}")
                return
        elif st.session_state["dataframe"] is not None:
            # Mostrar la vista previa del DataFrame ya procesado
            df = st.session_state["dataframe"]
            show_data_preview(df, "Guardado en memoria", len(st.session_state["csv_content"]))
        else:
            # Intentar procesar el CSV si está en la sesión pero el dataframe no
            try:
                df, encoding = process_csv(st.session_state["csv_content"])
                st.session_state["dataframe"] = df
                show_data_preview(df, encoding, len(st.session_state["csv_content"]))
            except Exception as e:
                st.error(f"Error al procesar el CSV guardado: {str(e)}")
                return

        # Si ya se generó la imagen, se muestra y se permite descargar sin perder el estado
        if st.session_state["generated_image"]:
            try:
                image_data = base64.b64decode(st.session_state["generated_image"])
                st.subheader("Resultado del Análisis")
                st.image(image_data, use_container_width=True, caption="Visualización Generada por IA")
                
                # Botones de acción bajo la imagen
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.download_button(
                        label="📥 Exportar como PNG",
                        data=image_data,
                        file_name="midas_plot.png",
                        mime="image/png"
                    )
                with col2:
                    if st.button("🔄 Nueva Visualización", help="Mantiene el dataset pero permite crear una nueva visualización"):
                        reset_visualization()
                
                # Mostrar código generado
                if st.session_state.get("generated_code"):
                    with st.expander("Código utilizado", expanded=False):
                        st.code(st.session_state["generated_code"], language="python")
                else:
                    with st.expander("Código utilizado", expanded=False):
                        st.warning("No se pudo recuperar el código generado")
            except Exception as e:
                st.error(f"Error al mostrar la imagen generada: {str(e)}")
        else:
            if st.button("🚀 Generar Visualización", use_container_width=True):
                with st.spinner("Generando gráfica, por favor espera..."):
                    df = st.session_state["dataframe"]
                    success, error_info = generate_visualization(df, st.session_state["user_prompt"])
                    
                    if success:
                        image_data = base64.b64decode(st.session_state["generated_image"])
                        st.subheader("Resultado del Análisis")
                        st.image(image_data, use_container_width=True, caption="Visualización Generada por IA")
                        
                        # Botones de acción bajo la imagen
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.download_button(
                                label="📥 Exportar como PNG",
                                data=image_data,
                                file_name="midas_plot.png",
                                mime="image/png"
                            )
                        with col2:
                            if st.button("🔄 Nueva Visualización", help="Mantiene el dataset pero permite crear una nueva visualización"):
                                reset_visualization()
                        
                        # Mostrar código generado
                        if st.session_state.get("generated_code"):
                            with st.expander("Código utilizado", expanded=False):
                                st.code(st.session_state["generated_code"], language="python")
                        else:
                            with st.expander("Código utilizado", expanded=False):
                                st.warning("No se pudo recuperar el código generado")
                    else:
                        st.error("Error al generar la visualización")
                        with st.expander("Detalles técnicos del error", expanded=True):
                            st.markdown('<div class="error-log">' + error_info["trace"] + '</div>', unsafe_allow_html=True)
                            
                            st.info("""
                            **Recomendaciones para solucionar este error:**
                            1. Intenta simplificar tu prompt y hacerlo más específico
                            2. Revisa si tu CSV tiene columnas con nombres complejos o caracteres especiales
                            3. Si el error persiste, prueba con un dataset más simple primero
                            """)
                            
                            # Mostrar información de diagnóstico
                            if "debug_info" in st.session_state:
                                with st.expander("Información de diagnóstico", expanded=False):
                                    st.json(st.session_state["debug_info"])
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
                2025 Midas Plot | Sistema Midas 
            </div>
        </footer>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
