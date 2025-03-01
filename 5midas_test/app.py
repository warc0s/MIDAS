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
with st.expander("ℹ️ ¿Cómo usar esta Midas Test?"):
    st.write("""
    **Pasos para analizar tu modelo:**
    1.  **Carga el archivo .joblib:** Utiliza el cargador de archivos para subir tu modelo de Machine Learning.
    2.  **Procesamiento del modelo:** La aplicación procesará el modelo.
    3.  **Iniciar evaluación con agentes:** Haz clic en el botón para que los agentes evalúen el modelo y generen un reporte.
    4.  **Espera 90 segundos:** Los agentes están trabajando en tu solicitud. Espera a que terminen.
    5.  **Finalizar análisis y descargar reporte:** Una vez pasado el tiempo, haz clic en este botón para ver los resultados y descargar el reporte en formato Markdown.
    """)

uploaded_file = st.file_uploader("", type=["joblib"])

if uploaded_file:
    file_path = "temp_model.joblib"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("⚙️ Procesando el modelo...")
    model_info = process_joblib(file_path)

    if "error" in model_info:
        st.error(model_info["error"])
    else:
        st.success("✅ Modelo analizado con éxito")
        # Botón para iniciar la evaluación con los agentes
        if st.button("🔄 Iniciar Evaluación con los Agentes"):
            group_manager.initiate_chat(
                recipient=group_manager,
                max_turns=3,  # Puedes ajustar este valor
                message=f"Analyze the following ML model: {model_info} and generate a Markdown report."
            )
            st.success("✅ Evaluación en proceso. Espera unos segundos...")

        # Botón para finalizar la evaluación y descargar el reporte
        if st.button("📄 Finalizar Análisis y Descargar Reporte"):
            if groupchat.messages:

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


                 # Gráfico de uso de CPU
                    fig_cpu, ax_cpu = plt.subplots()
                    ax_cpu.bar(['Uso de CPU'], [perf['cpu_usage']])
                    ax_cpu.set_ylabel('Porcentaje')
                    ax_cpu.set_title('Uso de CPU')
                    st.pyplot(fig_cpu)

                    # Gráfico de tiempo de predicción
                    fig_time, ax_time = plt.subplots()
                    ax_time.bar(['Tiempo de predicción'], [perf['prediction_time']])
                    ax_time.set_ylabel('Segundos')
                    ax_time.set_title('Tiempo de predicción')
                    st.pyplot(fig_time)

                    
                st.subheader("📄 Reporte Generado")
                report = generate_markdown_report(groupchat.messages)
                with open("informe_analisis_modelo.md", "r", encoding="utf-8") as f:
                    st.download_button("⬇️ Descargar Reporte", f, "informe_analisis_modelo.md", "text/markdown")
            else:
                st.warning("⚠️ No hay mensajes en el chat de grupo. No se generó reporte.")

    os.remove(file_path)  # Limpieza del archivo temporal