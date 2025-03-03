import streamlit as st
import os
from agents_test import process_joblib, generate_markdown_report, group_manager, groupchat

st.set_page_config(
    page_title="Midas Test | Midas System",
    page_icon="https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/test_trans.png?raw=true",
)

logo_url = "https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/test_trans.png?raw=true"

# Aplicar estilos personalizados
st.markdown(
    f"""
    <style>
        /* Cambiar el color de fondo */
        body {{
            background-color: #0A0F22;
        }}
        
        /* Cambiar el color del título */
        .title-container h1 {{
            color: #DAA520;
        }}
        
        /* Centrar el logo y el título */
        .title-container {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        /* Ajustar tamaño del logo */
        .title-container img {{
            width: 60px;
            height: 60px;
        }}

    </style>
    
    <div class="title-container">
        <img src="{logo_url}" alt="Logo">
        <h1>Midas Test</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### Análisis de Modelos de Machine Learning", unsafe_allow_html=True)

# Cuadrado de información sobre cómo usar la aplicación con fondo azul oscuro
st.markdown(
    """
    <style>
        details {
            background-color: #001F3F; /* Azul oscuro */
            color: white; /* Texto blanco para contraste */
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
        }

        details summary {
            color: white; /* Color del encabezado */
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Cuadrado de información sobre cómo usar la aplicación
with st.expander("ℹ️ ¿Cómo usar Midas Test?"):
    st.write("""
    **Pasos para analizar tu modelo:**
    1.  **Carga el archivo .joblib:** Utiliza el cargador de archivos para subir tu modelo de Machine Learning.
    2.  **Procesamiento del modelo:** La aplicación procesará el modelo.
    3.  **Iniciar evaluación con agentes:** Haz clic en el botón para que los agentes evalúen el modelo y generen un reporte.
    4.  **Espera 90 segundos:** Los agentes están trabajando en tu solicitud. Espera a que terminen.
    5.  **Finalizar análisis y descargar reporte:** Una vez pasado el tiempo, haz clic en este botón para ver los resultados y descargar el reporte en formato Markdown.
    """)

# Aplicar estilos personalizados a los botones
st.markdown(
    """
    <style>
        /* Estilo general de botones */
        .stButton>button {
            background-color: #001F3F !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 8px 15px !important;
            font-size: 16px !important;
            border: 2px solid transparent !important;
            transition: all 0.3s ease-in-out !important;
        }

        /* Cambiar color al pasar el mouse */
        .stButton>button:hover {
            background-color: #003fff !important;
            color: #0A0F22 !important;
            border: 2px solid white !important;
        }

        /* Estilo para botones deshabilitados */
        .stButton>button:disabled {
            background-color: #555555 !important;
            color: #AAAAAA !important;
            border: 2px solid #777777 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader("Insertar joblib: ", type=["joblib"], label_visibility="collapsed")



if uploaded_file:
    file_path = "temp_model.joblib"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("⚙️ Procesando el modelo...")
    model_info = process_joblib(file_path)

    if "error" in model_info:
        st.error(model_info["error"])
    else:
        st.success("✅ Modelo recibido con éxito")

        # Colocar ambos botones siempre visibles
        col1, col2 = st.columns(2)

        with col1:
            iniciar_evaluacion = st.button("🔄 Iniciar Evaluación con los Agentes")
        
        with col2:
            finalizar_analisis = st.button("📄 Finalizar Análisis y Descargar Reporte")

        # Acciones cuando se presionan los botones
        if iniciar_evaluacion:
            group_manager.initiate_chat(
                recipient=group_manager,
                max_turns=3,
                message=f"Analyze the following ML model: {model_info} and generate a Markdown report."
            )
            st.success("✅ Evaluación en proceso. Espera unos segundos...")

        if finalizar_analisis:
            if len(groupchat.messages) < 3 and not "See you soon!" in groupchat.messages:
                st.warning("⚠️ Los agentes siguen trabajando... Por favor, espera a que terminen.")
            else:
                st.subheader("📊 Resultados del Modelo")
                with st.expander("📌 Información del Modelo"):
                    st.write(f"**Tiempo de carga:** {model_info['load_time']:.4f} segundos")
                    st.write(f"**Tamaño en disco:** {model_info['size_on_disk']:.4f} MB")
                    st.write(f"**Uso de memoria:** {model_info['memory_usage']:.4f} MB")
                    st.write(f"**Rendimiento:** {model_info['throughput']:.4f} muestras/segundo")
                    st.write(f"**Recomendación final:** {model_info['final_recommendation']}")

                with st.expander("📈 Métricas de Rendimiento"):
                    perf = model_info['performance_metrics']
                    st.write(f"- **Pico de memoria:** {perf['memory_peak']}")
                    st.write(f"- **Uso de CPU:** {perf['cpu_usage']}%")
                    st.write(f"- **Tiempo de predicción:** {perf['prediction_time']:.4f} segundos")

                with st.expander("⚠️ Pruebas de Robustez"):
                    robustness = model_info['robustness_tests']
                    st.write(f"- **Valores nulos:** {robustness['null_values']}")
                    st.write(f"- **Fuera de rango:** {robustness['out_of_range']}")
                    st.write(f"- **Tipo de datos incorrecto:** {robustness['wrong_data_type']}")
                    st.write(f"- **Predicciones consistentes:** {robustness['consistent_predictions']}")

                st.subheader("📄 Reporte Generado")
                report = generate_markdown_report(groupchat.messages)
                with open("informe_analisis_modelo.md", "r", encoding="utf-8") as f:
                    st.download_button("⬇️ Descargar Reporte", f, "informe_analisis_modelo.md", "text/markdown")

    os.remove(file_path)  # Limpieza del archivo temporal