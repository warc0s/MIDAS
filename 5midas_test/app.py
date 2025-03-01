import streamlit as st
import os
from agents_test import process_joblib, generate_markdown_report, group_manager, groupchat

st.title("MIDAS TEST: Análisis de Modelos de Machine Learning")

uploaded_file = st.file_uploader("📂 Carga un archivo .joblib", type=["joblib"])

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
                    st.write(f"**Uso de memoria:** {model_info['memory_usage']:.2f} MB")
                    st.write(f"**Rendimiento:** {model_info['throughput']:.2f} muestras/segundo")
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
            else:
                st.warning("⚠️ No hay mensajes en el chat de grupo. No se generó reporte.")

    os.remove(file_path)  # Limpieza del archivo temporal
