import streamlit as st
import os
import zipfile
import uuid
import io
import time
import base64
from pathlib import Path
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr
import logging
import json
from dotenv import load_dotenv
import google.generativeai as genai
import shutil
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Importar el código de Midas Touch
from Midas_Touch_V2_5_CLI import (
    AICortex, WorkflowStage, CONFIG, setup_logging,
    BaseAgent, ModelShamanAgent, convert_to_serializable,
    OperationalContext
)

# Clase que extiende AICortex para trabajar con Streamlit
class StreamlitAICortex(AICortex):
    def __init__(self, session_id, uploaded_file=None):
        # Guardar ID de sesión
        self.session_id = session_id
        
        # Crear directorios para esta sesión
        self.session_dir = Path(f"sessions/{session_id}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear directorios necesarios para logs
        log_dir = self.session_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Modificar CONFIG para esta sesión
        global CONFIG
        self.original_config = CONFIG.copy()
        
        # Guardar configuración original para restaurarla después
        CONFIG.update({
            'LOG_FILE': str(log_dir / f"{session_id}_ml_workflow.log"),
            'DATA_FILE': str(self.session_dir / f"{session_id}_dataset.csv"),
            'MODEL_DIR': str(self.session_dir / f"{session_id}_models"),
            'NOTEBOOK_FILE': str(self.session_dir / f"{session_id}_ml_workflow_documentation.ipynb"),
        })
        
        # Si se subió un archivo, guardarlo en el directorio de la sesión
        if uploaded_file:
            self.save_uploaded_file(uploaded_file)
        
        # Inicializar la superclase con la configuración modificada
        super().__init__()
    
    def save_uploaded_file(self, uploaded_file):
        """Guarda el archivo subido en el directorio de la sesión"""
        file_path = Path(CONFIG['DATA_FILE'])
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    def execute_workflow(self, user_input: str):
        """Ejecuta el workflow completo y captura la salida"""
        # Configurar captura de logs
        log_output = io.StringIO()
        
        # Asegurarse de que el logger esté configurado para capturar todo
        logger = logging.getLogger()
        log_handler = logging.StreamHandler(log_output)
        log_handler.setLevel(logging.INFO)
        logger.addHandler(log_handler)
        
        # Suprimir advertencias específicas 
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        
        try:
            # Actualizar UI en tiempo real
            progress_placeholder = st.empty()
            
            # Fase 1: Carga de datos
            progress_placeholder.info("Cargando datos...")
            self.ctx.update_stage(WorkflowStage.DATA_LOADING)
            raw_data = self._load_data()
            self.ctx._state['dataset'] = raw_data

            # Fase 1.1: Análisis de intención
            progress_placeholder.info("Analizando la intención del usuario...")
            intent = self.agents['intent'].analyze_intent(user_input)
            
            # Guardar el tipo de problema en el contexto
            problem_type = intent.get('problem_type', 'classification')
            self.ctx.set_problem_type(problem_type)
            progress_placeholder.info(f"Tipo de problema detectado: {problem_type}")
            
            # Fase 1.2: Resolución de la columna objetivo
            progress_placeholder.info("Determinando la columna objetivo...")
            target_col = self.agents['data_guardian'].resolve_target(raw_data, intent)
            self.ctx.set_target_column(target_col)
            
            # Fase 2: Preprocesamiento adaptativo
            progress_placeholder.info(f"Iniciando preprocesamiento para predecir '{target_col}'...")
            clean_data, target = self.agents['data_alchemist'].auto_preprocess(raw_data)
            
            # Fase 3: Modelado inteligente
            progress_placeholder.info("Entrenando el modelo...")
            model = self.agents['model_shaman'].conjure_model(clean_data, target)
            
            # Fase 4: Validación final y deployment
            progress_placeholder.info("Validando y desplegando el modelo...")
            if self.agents['oracle'].validate_workflow():
                deployment_info = self._deploy_model(model)
                progress_placeholder.success("¡Workflow completado con éxito!")
                
                # Extraer logs del archivo
                log_content = ""
                log_file_path = Path(CONFIG['LOG_FILE'])
                if log_file_path.exists():
                    with open(log_file_path, 'r') as f:
                        log_content = f.read()
                
                return True, log_content or log_output.getvalue(), deployment_info
            else:
                progress_placeholder.warning("La validación del workflow ha fallado")
                return False, log_output.getvalue(), None
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            return False, f"{log_output.getvalue()}\n\n{error_traceback}", None
        
        finally:
            # Remover el handler de logs
            logger.removeHandler(log_handler)
    
    def _deploy_model(self, model):
        """Implementación mejorada de despliegue del modelo."""
        # Crear directorio para modelos si no existe
        model_dir = Path(CONFIG['MODEL_DIR'])
        model_dir.mkdir(exist_ok=True)
        
        # Generar timestamp para versionado
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}.joblib"
        model_path = model_dir / model_filename
        
        # Guardar el modelo
        import joblib
        joblib.dump(model, model_path)
        
        # Guardar metadatos si existen
        model_agents = [agent for agent in BaseAgent.instances 
                       if isinstance(agent, ModelShamanAgent)]
        
        if model_agents and hasattr(model_agents[0], 'metadata') and model_agents[0].metadata:
            metadata = model_agents[0].metadata
            metadata_path = model_dir / f"metadata_{timestamp}.json"
            
            # Convertir a diccionario para serialización JSON
            metadata_dict = vars(metadata)
            
            # Convertir valores no serializables a tipos nativos de Python
            serializable_metadata = convert_to_serializable(metadata_dict)
            
            # Guardar metadatos
            with open(metadata_path, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
        
        # Guardar un informe de rendimiento
        report_path = model_dir / f"report_{timestamp}.txt"
        metrics = convert_to_serializable(self.ctx.get_context()['performance_metrics'])
        problem_type = self.ctx.get_context().get('problem_type', 'classification')
        
        with open(report_path, 'w') as f:
            f.write(f"# Reporte de Modelo: {timestamp}\n\n")
            f.write(f"Columna objetivo: {self.ctx.get_context()['target_column']}\n")
            f.write(f"Tipo de problema: {problem_type}\n")
            f.write(f"Tipo de modelo: {model.__class__.__name__}\n\n")
            f.write("## Métricas de rendimiento\n\n")
            
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            # Añadir mapeo de clases si existe y es un problema de clasificación
            if problem_type == 'classification' and 'class_mapping' in self.ctx.get_context():
                f.write("\n## Mapeo de clases\n\n")
                class_mapping = self.ctx.get_context()['class_mapping']
                for key, value in class_mapping.items():
                    f.write(f"Clase {key}: {value}\n")
        
        self.ctx.update_stage(WorkflowStage.DEPLOYMENT)
        
        # Documentar el despliegue del modelo
        if hasattr(self.agents['notebook_scribe'], 'document_model_deployment'):
            self.agents['notebook_scribe'].document_model_deployment(str(model_path), metrics)
        
        return {
            'model_path': str(model_path),
            'report_path': str(report_path),
            'metrics': metrics,
            'notebook_path': CONFIG['NOTEBOOK_FILE'],
            'log_path': CONFIG['LOG_FILE'],
            'model_dir': str(model_dir),
            'problem_type': problem_type,
            'class_mapping': self.ctx.get_context().get('class_mapping', {})
        }
    
    def create_download_zip(self):
        """Crea un archivo ZIP con todos los archivos generados."""
        # Crear un archivo ZIP temporal
        zip_filename = f"{self.session_id}_results.zip"
        zip_path = self.session_dir / zip_filename
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Añadir el notebook
            notebook_path = Path(CONFIG['NOTEBOOK_FILE'])
            if notebook_path.exists():
                zipf.write(notebook_path, arcname=notebook_path.name)
            
            # Añadir logs
            log_path = Path(CONFIG['LOG_FILE'])
            if log_path.exists():
                zipf.write(log_path, arcname=log_path.name)
            
            # Añadir modelo y reportes
            model_dir = Path(CONFIG['MODEL_DIR'])
            if model_dir.exists():
                for file_path in model_dir.glob('*'):
                    zipf.write(file_path, arcname=f"models/{file_path.name}")
        
        return zip_path
    
    def cleanup(self):
        """Restaura la configuración original"""
        global CONFIG
        CONFIG = self.original_config


# Función para crear un enlace de descarga para un archivo
def get_download_link(file_path, text):
    if not os.path.exists(file_path):
        return "Archivo no disponible"
        
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'data:application/octet-stream;base64,{b64}'
    return f'<a href="{href}" download="{os.path.basename(file_path)}" class="download-button">{text}</a>'


# Función para mostrar una celda del notebook de forma más legible
def display_notebook_cell(cell):
    cell_type = cell.get('cell_type')
    source = ''.join(cell.get('source', []))
    
    if cell_type == 'markdown':
        # Mostrar en un contenedor con estilo controlado
        with st.container():
            st.markdown(f'<div class="notebook-markdown">{source}</div>', unsafe_allow_html=True)
    elif cell_type == 'code':
        # Mostrar código con scroll horizontal
        st.code(source, language='python')

# Aplicación Streamlit principal
def main():
    st.set_page_config(
        page_title="Midas Touch | Midas System",
        page_icon="https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/touch_trans.png?raw=true",
        layout="wide",
    )
    
    # CSS personalizado con colores universales compatibles con ambos modos
    st.markdown("""
    <style>
    /* Estilos del botón de descarga (universal) */
    .download-button {
        display: inline-block;
        padding: 8px 16px;
        background-color: #3498DB;
        color: white !important;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        border-radius: 4px;
        transition: background-color 0.3s;
        margin: 8px 0;
    }
    .download-button:hover {
        background-color: #2980B9;
        text-decoration: none;
    }

    /* Estilos para el formato del notebook */
    .notebook-markdown h1 {
        font-size: 1.5em !important;
    }
    .notebook-markdown h2 {
        font-size: 1.3em !important;
    }
    .notebook-markdown h3 {
        font-size: 1.1em !important;
    }

    /* Tarjetas de métricas (universal) */
    .metric-card {
        background-color: #48332f;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #3498DB;
    }
    .metric-label {
        font-size: 0.9em;
        color: #ECF0F1;
    }

    /* Tarjetas de descarga (universal) */
    .download-card {
        background-color: #2C3E50;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .download-title {
        text-align: center;
        margin-bottom: 10px;
        font-weight: bold;
        color: #ECF0F1;
    }
    .download-description {
        font-size: 0.85em;
        color: #BDC3C7;
        margin-bottom: 15px;
        flex-grow: 1;
    }
    
    /* Estilos para tabla de mapeo de clases */
    .class-mapping-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .class-mapping-table th, .class-mapping-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    .class-mapping-table th {
        background-color: #3498DB;
        color: white;
    }
    .class-mapping-table tr:nth-child(even) {
        background-color: rgba(52, 152, 219, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🪙 Midas Touch: Sistema Inteligente de Flujo de Trabajo ML")
    
    # Generar un ID de sesión único si no existe
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Inicialización
    if 'cortex' not in st.session_state:
        st.session_state.cortex = None
        st.session_state.workflow_completed = False
        st.session_state.logs = ""
        st.session_state.deployment_info = None
        st.session_state.dataframe = None
    
    with st.sidebar:
        st.subheader("Configuración")
        
        uploaded_file = st.file_uploader("Cargar archivo de datos", type=["csv", "xlsx", "parquet", "json"])
        
        # Cargar y mostrar vista previa del dataset
        if uploaded_file is not None:
            try:
                # Leer diferentes formatos
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                st.session_state.dataframe = df
            except Exception as e:
                st.error(f"Error al cargar el archivo: {str(e)}")
        
        # Input del usuario
        user_input = st.text_area("Describir tarea de ML", 
                               placeholder="Ej: 'predecir columna diabetes, problema de clasificación'",
                               height=100)
        
        # Mostrar ayuda sobre el formato esperado
        st.caption("Formato recomendado: 'predecir columna X, problema de clasificación/regresión'")
        
        # Botón para iniciar el procesamiento
        start_button = st.button("Iniciar Procesamiento", type="primary", 
                              disabled=(uploaded_file is None or not user_input))
        
        if start_button and uploaded_file is not None and user_input:
            # Configuración de la API de Gemini
            load_dotenv()
            api_key = os.getenv('google_key')
            if api_key:
                genai.configure(api_key=api_key)
            else:
                st.error("No se encontró la API key de Google. Asegúrate de configurar la variable de entorno 'google_key'.")
                st.stop()
            
            # Limpiar sesión anterior si existe
            if st.session_state.cortex:
                st.session_state.cortex.cleanup()
            
            # Crear una nueva instancia de StreamlitAICortex
            st.session_state.cortex = StreamlitAICortex(st.session_state.session_id, uploaded_file)
            
            with st.spinner("Procesando... Esto puede tomar varios minutos"):
                # Ejecutar workflow
                start_time = time.time()
                success, logs, deployment_info = st.session_state.cortex.execute_workflow(user_input)
                elapsed_time = time.time() - start_time
                
                st.session_state.workflow_completed = success
                st.session_state.logs = logs
                st.session_state.deployment_info = deployment_info
                st.session_state.elapsed_time = elapsed_time
            
            # Recargar la página para mostrar los resultados
            st.rerun()
    
    # Mostrar vista previa del dataset cargado
    if st.session_state.dataframe is not None:
        st.subheader("📊 Vista previa del dataset")
        with st.expander("Ver datos", expanded=True):
            # Limitar a 100 filas para no saturar la interfaz
            st.dataframe(st.session_state.dataframe.head(100), height=300)
            st.caption(f"Mostrando primeras 100 filas de {st.session_state.dataframe.shape[0]} filas, {st.session_state.dataframe.shape[1]} columnas")
            
            # Mostrar información de las columnas
            col_info = pd.DataFrame({
                'Tipo': st.session_state.dataframe.dtypes,
                'Valores únicos': st.session_state.dataframe.nunique(),
                'Valores faltantes': st.session_state.dataframe.isnull().sum(),
                '% Faltantes': (st.session_state.dataframe.isnull().sum() / len(st.session_state.dataframe) * 100).round(2)
            })
            st.markdown("##### Información de columnas")
            st.dataframe(col_info)
    
    # Mostrar resultados
    if st.session_state.workflow_completed:
        st.success(f"✅ Workflow completado con éxito en {st.session_state.elapsed_time:.2f} segundos!")
        
        # Mostrar información del problema
        problem_type = st.session_state.deployment_info.get('problem_type', 'classification')
        target_column = st.session_state.deployment_info.get('target_column', '')
        
        st.markdown(f"### 🎯 Tipo de problema: {problem_type.capitalize()}")
        
        # Mostrar mapeo de clases si es un problema de clasificación
        class_mapping = st.session_state.deployment_info.get('class_mapping', {})
        if problem_type == 'classification' and class_mapping:
            with st.expander("Ver mapeo de clases", expanded=True):
                st.markdown("#### Mapeo de Clases (Valor numérico → Etiqueta original)")
                
                # Crear tabla HTML para el mapeo de clases
                mapping_table = "<table class='class-mapping-table'><tr><th>Valor</th><th>Etiqueta original</th></tr>"
                for key, value in class_mapping.items():
                    mapping_table += f"<tr><td>{key}</td><td>{value}</td></tr>"
                mapping_table += "</table>"
                
                st.markdown(mapping_table, unsafe_allow_html=True)
        
        # Mostrar métricas
        metrics = st.session_state.deployment_info.get('metrics', {})
        st.subheader("📊 Métricas de rendimiento:")
        
        # Diseño de métricas más responsive
        if metrics:
            # Determinar el número de columnas basado en la cantidad de métricas
            num_cols = min(len(metrics), 4)  # Máximo 4 columnas
            
            # Distribuir métricas en filas
            metrics_items = list(metrics.items())
            metrics_chunks = [metrics_items[i:i + num_cols] for i in range(0, len(metrics_items), num_cols)]
            
            for chunk in metrics_chunks:
                cols = st.columns(num_cols)
                for i, (metric, value) in enumerate(chunk):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{value:.4f}</div>
                            <div class="metric-label">{metric}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Crear pestañas para los diferentes outputs
        tab1, tab2 = st.tabs(["📥 Descargas", "📝 Logs"])
                
        with tab1:
            # Sección de descargas rediseñada
            
            # Crear archivo ZIP con todos los resultados
            zip_path = st.session_state.cortex.create_download_zip()
            
            # Mostrar opciones de descarga en tarjetas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="download-card">
                    <div class="download-title">📦 Todo en uno</div>
                    <div class="download-description">Archivo ZIP completo con modelo, notebook, logs y reportes. Ideal para compartir todo el proyecto.</div>
                """, unsafe_allow_html=True)
                
                if os.path.exists(zip_path):
                    st.markdown(get_download_link(zip_path, "⬇️ Descargar ZIP completo"), unsafe_allow_html=True)
                else:
                    st.warning("ZIP no disponible")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="download-card">
                    <div class="download-title">📓 Notebook</div>
                    <div class="download-description">Jupyter Notebook con documentación detallada de todo el proceso, código reproducible y explicaciones.</div>
                """, unsafe_allow_html=True)
                
                notebook_path = st.session_state.deployment_info.get('notebook_path')
                if notebook_path and os.path.exists(notebook_path):
                    st.markdown(get_download_link(notebook_path, "⬇️ Descargar Notebook (.ipynb)"), unsafe_allow_html=True)
                else:
                    st.warning("Notebook no disponible")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="download-card">
                    <div class="download-title">🤖 Modelo entrenado</div>
                    <div class="download-description">Modelo de machine learning serializado, listo para usar en producción o para realizar predicciones.</div>
                """, unsafe_allow_html=True)
                
                model_path = st.session_state.deployment_info.get('model_path')
                if model_path and os.path.exists(model_path):
                    st.markdown(get_download_link(model_path, "⬇️ Descargar Modelo (.joblib)"), unsafe_allow_html=True)
                else:
                    st.warning("Modelo no disponible")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Añadir enlace para el reporte
            st.markdown("<br>", unsafe_allow_html=True)
            report_path = st.session_state.deployment_info.get('report_path')
            if report_path and os.path.exists(report_path):
                with st.expander("Ver reporte de rendimiento"):
                    with open(report_path, 'r') as f:
                        report_content = f.read()
                    st.text_area("", report_content, height=200)
                    st.markdown(get_download_link(report_path, "⬇️ Descargar Reporte (.txt)"), unsafe_allow_html=True)
        
        with tab2:
            # Mostrar logs
            st.markdown("### Logs del proceso")
            if st.session_state.logs:
                st.text_area("", st.session_state.logs, height=500)
                
                # Añadir botón para descargar logs
                log_path = st.session_state.deployment_info.get('log_path')
                if log_path and os.path.exists(log_path):
                    st.markdown(get_download_link(log_path, "⬇️ Descargar Logs completos"), unsafe_allow_html=True)
            else:
                st.warning("No hay logs disponibles")
    
    elif st.session_state.cortex is not None:
        # Si se inició el proceso pero falló
        st.error("❌ El workflow no se completó correctamente.")
        st.subheader("📝 Logs de error")
        st.text_area("", st.session_state.logs, height=400)
    
    elif st.session_state.dataframe is None:
        # Instrucciones iniciales
        st.info("👈 Sube un archivo de datos y describe tu tarea de ML en el panel lateral para comenzar.")
        
        # Mostrar una visión general del sistema
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Acerca de Midas Touch")
            st.markdown("""
            **Midas Touch** es un sistema inteligente de flujo de trabajo de Machine Learning que:
            
            1. **Analiza** automáticamente tus datos
            2. **Identifica** la mejor columna objetivo basada en tu descripción
            3. **Preprocesa** los datos aplicando transformaciones adecuadas
            4. **Entrena** un modelo optimizado para tu problema específico
            5. **Documenta** todo el proceso en un notebook reproducible
            
            Simplemente sube tu dataset y describe lo que quieres predecir.
            """)
        
        with col2:
            st.subheader("Ejemplo de uso")
            st.markdown("""
            1. **Sube** un archivo CSV con tus datos
            2. **Describe** tu tarea, por ejemplo:
               - "Predecir la columna 'precio' de las casas, problema de regresion"
               - "Clasificar clientes según la columna 'abandono', problema de clasificacion"
               - "Determinar si un correo es spam o no en la columna 'categoria', problema de clasificacion"
            3. **Inicia** el procesamiento y espera los resultados
            4. **Descarga** el modelo entrenado, notebook y documentación
            """)

if __name__ == "__main__":
    main()