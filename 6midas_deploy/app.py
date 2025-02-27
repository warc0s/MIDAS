import streamlit as st
import joblib
import os
from agents_deploy import process_joblib, start_conversation

def main():
    st.title("🔍 Machine Learning Model Interface")
    
    # Inicializar variables de sesión
    if "interface_ready" not in st.session_state:
        st.session_state.interface_ready = False
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""
    
    model_description = st.text_input("Describe brevemente para qué sirve el modelo:")
    uploaded_file = st.file_uploader("Sube un archivo .joblib con tu modelo", type=["joblib"])
    
    if uploaded_file is not None and model_description:
        file_path = "temp_model.joblib"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Iniciar generación de interfaz"):
            with st.spinner("Generando la interfaz... ⏳"):
                model_info = process_joblib(file_path)
                if "error" in model_info:
                    st.error(model_info["error"])
                else:
                    generated_code = start_conversation(model_info, model_description)  # Extraer código del agente
                    
                    if generated_code:
                        # Guardar el código en un archivo para garantizar su existencia
                        with open("generated_interface.py", "w", encoding="utf-8") as f:
                            f.write(generated_code)
                        
                        st.session_state.interface_ready = True
                        st.session_state.generated_code = generated_code
                        st.success("✅ ¡Interfaz generada correctamente!")

    if st.session_state.interface_ready:
        st.write("---")
        
        if os.path.exists("generated_interface.py"):
            with open("generated_interface.py", "r", encoding="utf-8") as f:
                st.session_state.generated_code = f.read()

            st.download_button(
                label="📥 Descargar código generado",
                data=st.session_state.generated_code,
                file_name="generated_interface.py",
                mime="text/plain"
            )
            
            if st.button("🚀 Ver código"):
                st.code(st.session_state.generated_code, language="python")
                st.success("Para desplegar, guarda el código en un archivo y ejecuta: `streamlit run generated_interface.py`")
        else:
            st.error("⚠️ El código aún no ha sido generado. Inténtalo de nuevo después de la generación.")
    
if __name__ == "__main__":
    main()
