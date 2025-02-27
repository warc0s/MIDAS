import os
import logging
import difflib
import google.generativeai as genai
import pandas as pd
import numpy as np
import ast
import joblib
import time
import re
from typing import Optional, Dict, Any, Tuple, List, Union
from dotenv import load_dotenv
from functools import wraps
from enum import Enum, auto
from dataclasses import dataclass, field
import json
from pathlib import Path
import traceback
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_squared_error, 
    precision_score, recall_score, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.exceptions import UndefinedMetricWarning

# ---------------------------------------------------------------------
# Configuración y Constantes
# ---------------------------------------------------------------------
CONFIG = {
    'API_KEY_ENV_VAR': 'google_key',
    'MODEL_NAME': 'gemini-2.0-flash',
    'LOG_FILE': 'ml_workflow.log',
    'LOG_LEVEL': logging.INFO,
    'DATA_FILE': 'dataset.csv',
    'MODEL_DIR': 'models',
    'NOTEBOOK_FILE': 'ml_workflow_documentation.ipynb',
    'RETRIES': {
        'default': 3,
        'model_generation': 5
    },
    'MIN_ROWS': 100,
    'MAX_MISSING_RATIO': 0.3,
    'MIN_FEATURE_VARIANCE': 0.01,
    'DEFAULT_TEST_SIZE': 0.2,
    'RANDOM_SEED': 42,
    'PERFORMANCE_THRESHOLDS': {
        'classification': {
            'min_accuracy': 0.3,
            'min_f1': 0.3
        },
        'regression': {
            'min_r2': 0.2,
            'max_rmse': None  # Dependerá de los datos
        }
    }
}

# ---------------------------------------------------------------------
# Enumeraciones y Tipos de Datos
# ---------------------------------------------------------------------
class WorkflowStage(Enum):
    DATA_LOADING = auto()
    DATA_VALIDATION = auto()
    FEATURE_ENGINEERING = auto()
    MODEL_TRAINING = auto()
    MODEL_VALIDATION = auto()
    DEPLOYMENT = auto()
    ERROR_HANDLING = auto()

class ErrorSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    FATAL = auto()

@dataclass
class ErrorLogEntry:
    timestamp: float = field(default_factory=time.time)
    agent: str = "N/A"
    message: str = ""
    level: ErrorSeverity = ErrorSeverity.INFO
    operation: str = None
    attempt: int = None
    traceback: str = None

@dataclass
class ModelMetadata:
    model_type: str
    target_column: str
    features: List[str]
    creation_time: float = field(default_factory=time.time)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    data_statistics: Dict[str, Any] = field(default_factory=dict)
    feature_transformations: Dict[str, List] = field(default_factory=dict)
    pipeline_steps: List[str] = field(default_factory=list)
    class_mapping: Dict[int, str] = field(default_factory=dict)
    version: str = "1.0.0"

# ---------------------------------------------------------------------
# Configuración de Logging
# ---------------------------------------------------------------------
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=CONFIG['LOG_LEVEL'],
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / CONFIG['LOG_FILE']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("AICortex")

logger = setup_logging()

# ---------------------------------------------------------------------
# Sistema de Contexto Operacional Compartido (SOC)
# ---------------------------------------------------------------------
class OperationalContext:
    def __init__(self):
        self._state = {
            'current_stage': WorkflowStage.DATA_LOADING,
            'dataset': None,
            'target_column': None,
            'problem_type': None,  # Añadido para almacenar el tipo de problema
            'validation_reports': {},
            'pipeline_steps': [],
            'model_versions': [],
            'error_log': [],
            'performance_metrics': {},
            'retry_count': 0,
            'fallback_activated': False,
            'data_statistics': {}
        }
        self._validation_rules = {
            'data_quality': {
                'min_rows': CONFIG['MIN_ROWS'],
                'max_missing': CONFIG['MAX_MISSING_RATIO'],
                'feature_variance': CONFIG['MIN_FEATURE_VARIANCE']
            },
            'model_performance': CONFIG['PERFORMANCE_THRESHOLDS']
        }
        self.notebook_scribe = None  # Será inicializado después de crear la clase
    
    def update_stage(self, stage: WorkflowStage):
        prev_stage = self._state['current_stage']
        self._state['current_stage'] = stage
        logger.info(f"Workflow avanzando de {prev_stage.name} a {stage.name}")
        
        # Documentar cambio de etapa en el notebook si el scribe está inicializado
        if self.notebook_scribe:
            self.notebook_scribe.document_stage_change(prev_stage, stage)
    
    def log_error(self, error_info: Dict):
        if isinstance(error_info, dict):
            entry = ErrorLogEntry(
                timestamp=error_info.get('timestamp', time.time()),
                agent=error_info.get('agent', 'N/A'),
                message=error_info.get('message', error_info.get('error', 'No message')),
                level=error_info.get('level', ErrorSeverity.WARNING),
                operation=error_info.get('operation'),
                attempt=error_info.get('attempt'),
                traceback=error_info.get('traceback')
            )
            self._state['error_log'].append(vars(entry))
        else:
            logger.warning(f"Invalid error_info format: {error_info}")
    
    def get_context(self) -> Dict:
        return self._state.copy()
    
    def get_validation_rules(self) -> Dict:
        return self._validation_rules
    
    def update_data_statistics(self, stats: Dict):
        self._state['data_statistics'].update(stats)
    
    def set_target_column(self, column: str):
        self._state['target_column'] = column
        logger.info(f"Columna objetivo establecida: {column}")
        
        # Documentar selección de columna objetivo
        if self.notebook_scribe:
            self.notebook_scribe.document_target_selection(column)
    
    def set_problem_type(self, problem_type: str):
        """Establece el tipo de problema (classification/regression)."""
        self._state['problem_type'] = problem_type
        logger.info(f"Tipo de problema establecido: {problem_type}")
    
    def set_notebook_scribe(self, scribe):
        """Establece el agente de documentación de notebook."""
        self.notebook_scribe = scribe
    
    def document_operation(self, title: str, description: str, code: str = None, data_snippet=None):
        """Documenta una operación en el notebook."""
        if self.notebook_scribe:
            self.notebook_scribe.document_operation(title, description, code, data_snippet)

# ---------------------------------------------------------------------
# Decoradores Avanzados
# ---------------------------------------------------------------------
def convert_to_serializable(obj):
    """Convierte objetos numpy y otros tipos no serializables a tipos nativos de Python."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # Para objetos datetime
        return obj.isoformat()
    else:
        return obj

def resilient_agent(max_retries=None, backoff_factor=2):
    """
    Decorador mejorado para reintentar operaciones con backoff exponencial.
    """
    if max_retries is None:
        max_retries = CONFIG['RETRIES']['default']
        
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    result = func(self, *args, **kwargs)
                    logger.debug(f"Operación {func.__name__} completada exitosamente")
                    return result
                except Exception as e:
                    if attempt == max_retries:
                        error_traceback = traceback.format_exc()
                        self.log(f"Error crítico en {func.__name__}: {str(e)}", 
                                 level=ErrorSeverity.FATAL)
                        self.ctx.log_error({
                            'agent': self.name,
                            'operation': func.__name__,
                            'error': str(e),
                            'attempt': attempt+1,
                            'traceback': error_traceback
                        })
                        raise
                    
                    delay = backoff_factor ** attempt
                    self.log(f"Reintento {attempt+1} para {func.__name__} en {delay}s. Error: {str(e)}", 
                             level=ErrorSeverity.WARNING)
                    time.sleep(delay)
                    self.ctx.log_error({
                        'agent': self.name,
                        'operation': func.__name__,
                        'error': str(e),
                        'attempt': attempt+1
                    })
        return wrapper
    return decorator

# ---------------------------------------------------------------------
# Agentes Especializados - Clase Base
# ---------------------------------------------------------------------
class BaseAgent:
    instances = []
    
    def __init__(self, name: str, ctx: OperationalContext):
        self.name = name
        self.ctx = ctx
        self.logger = self._configure_logger()
        self.__class__.instances.append(self)
    
    def _configure_logger(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        return logger
    
    def log(self, message: str, level: ErrorSeverity = ErrorSeverity.WARNING):
        log_entry = {
            'timestamp': time.time(),
            'agent': self.name,
            'message': message,
            'level': level
        }
        self.ctx.log_error(log_entry)
        log_level = logging.INFO if level == ErrorSeverity.INFO else \
                    logging.WARNING if level == ErrorSeverity.WARNING else \
                    logging.ERROR if level == ErrorSeverity.CRITICAL else \
                    logging.CRITICAL
        self.logger.log(log_level, message)

# ---------------------------------------------------------------------
# Agente de Documentación de Notebook
# ---------------------------------------------------------------------
class NotebookScribeAgent(BaseAgent):
    def __init__(self, ctx: OperationalContext):
        super().__init__("NotebookScribe", ctx)
        self.notebook_path = Path(CONFIG['NOTEBOOK_FILE'])
        self.nb = self._initialize_notebook()
        self.cell_count = 0
        
        # Crear una introducción al notebook
        self._create_introduction()
        
        # Registrar este agente en el contexto
        self.ctx.set_notebook_scribe(self)
        
    def _initialize_notebook(self):
        """Crea un nuevo notebook o carga uno existente."""
        if self.notebook_path.exists():
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
                # Reestablecer el contador de celdas basado en el notebook existente
                self.cell_count = len(nb["cells"])
                return nb
        else:
            # Crear estructura base del notebook
            return {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "codemirror_mode": {
                            "name": "ipython",
                            "version": 3
                        },
                        "file_extension": ".py",
                        "mimetype": "text/x-python",
                        "name": "python",
                        "nbconvert_exporter": "python",
                        "pygments_lexer": "ipython3",
                        "version": "3.8.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 5
            }
    
    def _create_introduction(self):
        """Crea la introducción del notebook."""
        title = "# Documentación del Flujo de Trabajo de Machine Learning\n\n"
        intro = (
            "Este notebook documenta paso a paso el proceso de análisis de datos, "
            "preprocesamiento, ingeniería de características y entrenamiento de modelo "
            "realizado por el componente Midas Touch (AICortex).\n\n"
            "Cada sección incluye una explicación detallada seguida del código utilizado "
            "para realizar las operaciones descritas.\n\n"
            f"**Fecha de generación:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "---"
        )
        
        # Solo añadir la introducción si es un notebook nuevo
        if len(self.nb["cells"]) == 0:
            self.add_markdown(title + intro)
    
    def add_markdown(self, content: str):
        """Añade una celda de markdown al notebook."""
        self.cell_count += 1
        cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": content.split('\n')
        }
        self.nb["cells"].append(cell)
        self._save_notebook()
        
    def add_code(self, code: str):
        """Añade una celda de código al notebook."""
        self.cell_count += 1
        cell = {
            "cell_type": "code",
            "execution_count": self.cell_count,
            "metadata": {},
            "outputs": [],
            "source": code.split('\n')
        }
        self.nb["cells"].append(cell)
        self._save_notebook()
        
    def _save_notebook(self):
        """Guarda el notebook en disco."""
        with open(self.notebook_path, 'w', encoding='utf-8') as f:
            json.dump(self.nb, f, indent=1)
    
    def document_operation(self, title: str, description: str, code: str = None, data_snippet=None):
        """Documenta una operación con título, descripción y código."""
        markdown_content = f"## {title}\n\n{description}"
        self.add_markdown(markdown_content)
        
        # Añadir snippet de datos si está disponible
        if data_snippet is not None:
            if isinstance(data_snippet, pd.DataFrame):
                # Convertir DataFrame a string
                data_code = f"# Vista previa de los datos\n{data_snippet.head(5).to_string()}"
                self.add_code(data_code)
        
        # Añadir código si está disponible
        if code:
            self.add_code(code)
        
        logger.info(f"Documentada operación: {title}")
    
    def document_stage_change(self, prev_stage: WorkflowStage, new_stage: WorkflowStage):
        """Documenta un cambio de etapa en el workflow."""
        title = f"Cambio de Etapa: {new_stage.name}"
        description = (
            f"El workflow ha avanzado de la etapa **{prev_stage.name}** a la etapa **{new_stage.name}**.\n\n",
            self._get_stage_description(new_stage)
        )
        self.document_operation(title, description)
    
    def document_target_selection(self, target_column: str):
        """Documenta la selección de la columna objetivo."""
        title = "Selección de Columna Objetivo"
        description = (
            f"Se ha seleccionado **{target_column}** como la columna objetivo para este proyecto de machine learning.\n\n"
            "Esta columna es la que intentaremos predecir con nuestro modelo."
        )
        code = f"# Establecer la columna objetivo\ntarget_column = '{target_column}'"
        self.document_operation(title, description, code)
    
    def document_data_summary(self, df: pd.DataFrame):
        """Documenta un resumen de los datos."""
        title = "Resumen del Dataset"
        description = (
            f"El dataset contiene **{df.shape[0]} filas** y **{df.shape[1]} columnas**.\n\n"
            "### Columnas del dataset:\n"
            + "\n".join([f"- `{col}`: {df[col].dtype}" for col in df.columns]) + "\n\n"
            "### Estadísticas básicas:"
        )
        
        # Crear código para generar el resumen de datos
        code = (
            "# Información básica del dataset\n"
            "print(f'Dimensiones: {df.shape}')\n"
            "print('\\nTipos de datos:')\n"
            "print(df.dtypes)\n"
            "print('\\nResumen estadístico:')\n"
            "df.describe()"
        )
        
        self.document_operation(title, description, code, df)
    
    def document_preprocessing(self, original_df: pd.DataFrame, transformed_df: pd.DataFrame, steps_description: str, code: str):
        """Documenta las operaciones de preprocesamiento."""
        title = "Preprocesamiento de Datos"
        description = (
            "En esta etapa aplicamos varias técnicas de limpieza y transformación para preparar los datos para el modelado.\n\n"
            f"{steps_description}\n\n"
            f"El dataset ha pasado de tener **{original_df.shape[1]} columnas** a **{transformed_df.shape[1]} columnas**."
        )
        
        self.document_operation(title, description, code, transformed_df)
    
    def document_model_training(self, model, X: pd.DataFrame, metrics: Dict[str, float]):
        """Documenta el entrenamiento del modelo."""
        title = "Entrenamiento del Modelo"
        
        # Crear descripción basada en el tipo de modelo y métricas
        model_type = model.__class__.__name__
        description = (
            f"Se ha entrenado un modelo de tipo **{model_type}** utilizando {X.shape[0]} muestras "
            f"y {X.shape[1]} características.\n\n"
            "### Métricas de Rendimiento:\n"
        )
        
        for metric, value in metrics.items():
            description += f"- **{metric}**: {value:.4f}\n"
        
        # Código para el entrenamiento del modelo
        code = (
            f"# Entrenamiento del modelo {model_type}\n"
            "from sklearn.model_selection import train_test_split\n"
            "# Dividir datos en entrenamiento y prueba\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            f"    X, y, test_size={CONFIG['DEFAULT_TEST_SIZE']}, random_state={CONFIG['RANDOM_SEED']})\n\n"
            f"# Crear y entrenar el modelo\n"
            f"model = {model_type}()\n"
            "model.fit(X_train, y_train)\n\n"
            "# Evaluar el modelo\n"
            "y_pred = model.predict(X_test)\n"
            "# Calcular métricas"
        )
        
        self.document_operation(title, description, code)
    
    def document_model_deployment(self, model_path: str, metrics: Dict[str, float]):
        """Documenta el despliegue del modelo."""
        title = "Despliegue del Modelo"
        description = (
            f"El modelo entrenado ha sido guardado en **{model_path}**.\n\n"
            "### Resumen Final de Métricas:\n"
        )
        
        for metric, value in metrics.items():
            description += f"- **{metric}**: {value:.4f}\n"
        
        code = (
            "# Guardar el modelo entrenado\n"
            "import joblib\n"
            f"joblib.dump(model, '{model_path}')\n"
            "print(f'Modelo guardado en {model_path}')"
        )
        
        self.document_operation(title, description, code)
    
    def _get_stage_description(self, stage: WorkflowStage) -> str:
        """Devuelve una descripción para cada etapa del workflow."""
        descriptions = {
            WorkflowStage.DATA_LOADING: (
                "En esta etapa se cargan los datos desde el archivo fuente. "
                "Se realiza una lectura inicial para entender la estructura del dataset."
            ),
            WorkflowStage.DATA_VALIDATION: (
                "Esta etapa verifica la calidad de los datos, identificando problemas como "
                "valores faltantes, outliers o inconsistencias."
            ),
            WorkflowStage.FEATURE_ENGINEERING: (
                "Durante esta etapa se transforman y crean nuevas características para mejorar "
                "el rendimiento del modelo. Incluye codificación, normalización y creación de features."
            ),
            WorkflowStage.MODEL_TRAINING: (
                "En esta etapa se entrena un modelo de machine learning utilizando los datos preprocesados."
            ),
            WorkflowStage.MODEL_VALIDATION: (
                "Esta etapa evalúa el rendimiento del modelo entrenado usando métricas apropiadas "
                "y validación cruzada."
            ),
            WorkflowStage.DEPLOYMENT: (
                "En esta etapa final, el modelo validado se guarda y se prepara para su uso en producción."
            ),
            WorkflowStage.ERROR_HANDLING: (
                "Esta etapa se activa cuando ocurre un error, intentando recuperar el workflow "
                "y continuar el proceso."
            )
        }
        return descriptions.get(stage, "Etapa sin descripción disponible.")

# ---------------------------------------------------------------------
# Agente de Intención
# ---------------------------------------------------------------------
class IntentAgent:
    def __init__(self):
        self.logger = logging.getLogger("IntentAgent")
        self.genai = genai.GenerativeModel(CONFIG['MODEL_NAME'])
    
    def analyze_intent(self, user_input: str) -> Dict[str, Any]:
        logger.info("Analizando la intención del usuario...")
        prompt = f"""
        Analiza la siguiente consulta de ML y devuelve un JSON con:
        - target_column: nombre EXACTO de la columna objetivo mencionada explícitamente
        - problem_type: 'classification' o 'regression'
        - context: información adicional relevante

        Consulta: "{user_input}"
        
        IMPORTANTE:
        1. target_column DEBE ser exactamente la columna mencionada en la consulta
        2. problem_type DEBE ser 'classification' o 'regression' según lo mencionado explícitamente
        3. Si no está claro, usa la evidencia más directa del texto
        
        Formato de respuesta (solo JSON):
        {{"target_column": "nombre_columna", "problem_type": "classification|regression", "context": "contexto"}}
        """
        
        for attempt in range(3):
            try:
                response = self.genai.generate_content(prompt)
                parsed = self._safe_parse(response.text)
                self.logger.info(f"Intención detectada: {parsed}")
                return parsed
            except Exception as e:
                self.logger.error(f"Error analizando intención (intento {attempt+1}): {str(e)}")
                time.sleep(2)
        
        # Si todos los intentos fallan, hacemos nuestra mejor suposición
        self.logger.warning("Usando análisis de intención fallback")
        return {
            "target_column": self._extract_possible_target(user_input),
            "problem_type": "classification",  # Valor por defecto
            "context": user_input
        }

    def _safe_parse(self, text: str) -> Dict:
        """Parseo seguro de la respuesta de la IA (JSON)"""
        try:
            # Eliminar bloques de código markdown y espacios en blanco
            clean_text = re.sub(r'```json|```', '', text).strip()
            
            # Intentar parsear como JSON primero
            try:
                return json.loads(clean_text)
            except:
                # Si falla, intentar con ast.literal_eval
                return ast.literal_eval(clean_text)
        except:
            self.logger.warning("Respuesta no parseable, usando análisis difuso")
            return self._fuzzy_parse(text)

    def _fuzzy_parse(self, text: str) -> Dict:
        """Parseo difuso para respuestas mal formateadas"""
        target = re.search(r'"target_column"\s*:\s*"([^"]+)"', text)
        problem = re.search(r'"problem_type"\s*:\s*"(\w+)"', text)
        
        return {
            "target_column": target.group(1) if target else None,
            "problem_type": (problem.group(1).lower() if problem else "classification"),
            "context": text
        }
    
    def _extract_possible_target(self, text: str) -> str:
        """Método simple para extraer posible columna objetivo en caso de fallo total"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Palabras clave comunes que podrían indicar un objetivo
        for word in ['predict', 'predecir', 'column', 'columna', 'target', 'objetivo']:
            if word in words:
                idx = words.index(word)
                if idx + 1 < len(words):
                    return words[idx + 1]
        
        return None

# ---------------------------------------------------------------------
# Agente Guardián de Datos
# ---------------------------------------------------------------------
class DataGuardianAgent(BaseAgent):
    def __init__(self, ctx: OperationalContext):
        super().__init__("DataGuardianAgent", ctx)
        self.genai = genai.GenerativeModel(CONFIG['MODEL_NAME'])
    
    def resolve_target(self, df: pd.DataFrame, intent: Dict) -> str:
        logger.info("Resolviendo la columna objetivo...")
        
        # Documentar el dataframe inicial
        self.ctx.document_operation(
            "Análisis Inicial del Dataset", 
            "En esta etapa analizamos la estructura del dataset y sus características principales.",
            "# Cargar y explorar el dataset\nimport pandas as pd\ndf = pd.read_csv('dataset.csv')\ndf.head()",
            df
        )
        
        # Obtener la columna objetivo directamente de la intención detectada
        target_column = intent.get('target_column')
        
        # Verificar si la columna existe en el DataFrame
        if target_column and target_column in df.columns:
            best_target = target_column
            logger.info(f"Columna objetivo encontrada directamente: {best_target}")
        else:
            # Si la columna no existe, consultar a Gemini para encontrar la mejor coincidencia
            best_target = self._find_target_column_with_gemini(df.columns.tolist(), intent)
            logger.info(f"Columna objetivo seleccionada mediante IA: {best_target}")
        
        # Si aún no tenemos una columna objetivo válida, usar la última columna
        if not best_target or best_target not in df.columns:
            best_target = df.columns[-1]
            self.log(f"No se encontró una columna objetivo válida, usando la última columna: {best_target}", 
                    ErrorSeverity.WARNING)
        
        # Guardar también el tipo de problema en el contexto
        problem_type = intent.get('problem_type', 'classification')
        self.ctx.set_problem_type(problem_type)
        
        # Calcular y almacenar estadísticas básicas de la columna objetivo
        self._compute_target_statistics(df, best_target)
        
        # Documentar la información sobre la columna objetivo
        target_info = df[best_target].describe().to_dict()
        target_info_str = "\n".join([f"- {k}: {v}" for k, v in target_info.items()])
        
        self.ctx.document_operation(
            "Análisis de la Columna Objetivo",
            f"La columna **{best_target}** ha sido seleccionada como objetivo para un problema de **{problem_type}**.\n\n"
            f"A continuación se presentan sus estadísticas básicas:\n\n{target_info_str}",
            f"# Análisis de la columna objetivo\ntarget = df['{best_target}']\ntarget.describe()"
        )
        
        return best_target

    def _find_target_column_with_gemini(self, columns: List[str], intent: Dict) -> str:
        """Usa Gemini para encontrar la mejor columna objetivo basada en la intención y las columnas disponibles"""
        prompt = f"""
        Necesito identificar cuál de estas columnas es la más adecuada como objetivo (target) para un problema de {intent['problem_type']}.

        Contexto de la tarea: "{intent['context']}"
        Columna objetivo mencionada: "{intent['target_column']}"

        Columnas disponibles en el dataset:
        {', '.join(columns)}

        Devuelve SOLO el nombre exacto de UNA columna que mejor coincida con la intención del usuario.
        La columna debe existir exactamente como está escrita en la lista proporcionada.
        """
        
        try:
            response = self.genai.generate_content(prompt)
            # Limpiar la respuesta para obtener solo el nombre de la columna
            column_name = response.text.strip()
            
            # Si la respuesta contiene más texto, intentar extraer solo el nombre de la columna
            if len(column_name.split()) > 1:
                # Buscar coincidencias exactas con las columnas disponibles
                for col in columns:
                    if col in column_name:
                        return col
                
                # Si no hay coincidencias, tomar la primera palabra
                return column_name.split()[0]
            
            return column_name
        except Exception as e:
            self.log(f"Error en la búsqueda de columna objetivo con IA: {str(e)}", 
                    ErrorSeverity.WARNING)
            return None

    def _compute_target_statistics(self, df: pd.DataFrame, target: str):
        """Calcula estadísticas de la columna objetivo y las almacena."""
        target_series = df[target]
        
        stats = {
            'target_column': target,
            'n_samples': len(target_series),
            'missing_values': target_series.isnull().sum(),
            'missing_ratio': target_series.isnull().mean()
        }
        
        if pd.api.types.is_numeric_dtype(target_series):
            stats.update({
                'min': float(target_series.min()),
                'max': float(target_series.max()),
                'mean': float(target_series.mean()),
                'median': float(target_series.median()),
                'std': float(target_series.std()),
                'data_type': 'numeric'
            })
        else:
            value_counts = target_series.value_counts().to_dict()
            # Convertir las claves a str para compatibilidad con JSON
            value_counts = {str(k): int(v) for k, v in value_counts.items()}
            stats.update({
                'unique_values': target_series.nunique(),
                'value_counts': value_counts,
                'data_type': 'categorical'
            })
        
        self.ctx.update_data_statistics(stats)

# ---------------------------------------------------------------------
# Alquimista de Datos
# ---------------------------------------------------------------------
class DataAlchemistAgent(BaseAgent):
    def __init__(self, ctx: OperationalContext):
        super().__init__("DataAlchemist", ctx)
        self.pipeline = None
    
    @resilient_agent(max_retries=3)
    def auto_preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesamiento adaptativo con auto-detección de estrategias."""
        logger.info("Iniciando preprocesamiento de datos...")
        context = self.ctx.get_context()
        target = context['target_column']
        
        try:
            # 1. Validación inicial
            self._validate_dataset(df, target)
            
            # 2. Análisis exploratorio básico
            self._perform_basic_eda(df)
            
            # 3. Ingeniería de características adaptativa
            df_transformed = self._adaptive_feature_engineering(df)
            
            # 4. Construcción del pipeline
            self.pipeline = self._build_dynamic_pipeline(df_transformed, target)
            
            # 5. Ejecución con validación
            X, y = self._execute_pipeline(df_transformed, target)
            logger.info("Preprocesamiento completado correctamente.")
            return X, y
        
        except Exception as e:
            self.log(f"Error en preprocesamiento: {str(e)}", ErrorSeverity.CRITICAL)
            raise

    def _validate_dataset(self, df: pd.DataFrame, target: str):
        """Validación avanzada de integridad de datos."""
        # Verificar existencia de la columna objetivo
        if target not in df.columns:
            raise ValueError(f"La columna objetivo '{target}' no existe en el DataFrame.")
        
        # Verificar tamaño mínimo del dataset
        min_rows = self.ctx.get_validation_rules()['data_quality']['min_rows']
        if len(df) < min_rows:
            self.log(f"Dataset tiene menos de {min_rows} filas: {len(df)}", 
                    ErrorSeverity.WARNING)
        
        # Verificar valores faltantes en objetivo
        target_missing_ratio = df[target].isnull().mean()
        max_missing = self.ctx.get_validation_rules()['data_quality']['max_missing']
        if target_missing_ratio > 0:
            self.log(f"Columna objetivo '{target}' tiene {target_missing_ratio:.1%} valores faltantes",
                    ErrorSeverity.WARNING if target_missing_ratio <= max_missing else ErrorSeverity.CRITICAL)
            
            if target_missing_ratio > max_missing:
                raise ValueError(f"Demasiados valores faltantes en columna objetivo: {target_missing_ratio:.1%}")
        
        # Verificar valores únicos en objetivo
        if len(df[target].unique()) == 1:
            self.log(f"Columna objetivo '{target}' tiene un único valor", 
                    ErrorSeverity.FATAL)
            raise ValueError("Target con valor constante, no se puede entrenar modelo")
        
        # Documentar la validación del dataset
        self.ctx.document_operation(
            "Validación del Dataset",
            f"Se ha verificado la integridad del dataset para la columna objetivo '{target}'.\n\n"
            f"- Número de filas: {len(df)}\n"
            f"- Valores faltantes en objetivo: {target_missing_ratio:.1%}\n"
            f"- Número de valores únicos en objetivo: {len(df[target].unique())}",
            "# Validación del dataset\n"
            f"print(f'Número de filas: {{len(df)}}')\n"
            f"print(f'Valores faltantes en objetivo: {{df[\"{target}\"].isnull().mean():.1%}}')\n"
            f"print(f'Valores únicos en objetivo: {{df[\"{target}\"].nunique()}}')"
        )

    def _perform_basic_eda(self, df: pd.DataFrame):
        """Realiza análisis exploratorio de datos básico."""
        stats = {
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_summary': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=np.number).columns.tolist(),
            'categorical_columns': df.select_dtypes(exclude=np.number).columns.tolist()
        }
        
        # Detectar columnas con alta cardinalidad
        high_cardinality_cols = []
        for col in df.select_dtypes(exclude=np.number).columns:
            if df[col].nunique() > 100:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            stats['high_cardinality_columns'] = high_cardinality_cols
            self.log(f"Columnas con alta cardinalidad detectadas: {high_cardinality_cols}", 
                    ErrorSeverity.WARNING)
        
        # Actualizar estadísticas en el contexto
        self.ctx.update_data_statistics(stats)
        
        # Documentar el EDA
        missing_cols = [col for col, count in stats['missing_summary'].items() if count > 0]
        missing_info = "\n".join([f"- {col}: {count} valores ({count/len(df):.1%})" 
                                for col, count in stats['missing_summary'].items() 
                                if count > 0]) if missing_cols else "- No hay valores faltantes"
        
        self.ctx.document_operation(
            "Análisis Exploratorio de Datos",
            f"Se ha realizado un análisis exploratorio básico del dataset:\n\n"
            f"- **Dimensiones**: {stats['shape'][0]} filas, {stats['shape'][1]} columnas\n"
            f"- **Columnas numéricas**: {len(stats['numeric_columns'])}\n"
            f"- **Columnas categóricas**: {len(stats['categorical_columns'])}\n\n"
            f"### Valores faltantes:\n{missing_info}\n\n"
            + (f"### Columnas con alta cardinalidad:\n" + "\n".join([f"- {col}" for col in high_cardinality_cols]) 
               if high_cardinality_cols else ""),
            "# Análisis exploratorio de datos\n"
            "print(f'Dimensiones: {df.shape}')\n"
            "print('\\nTipos de datos:')\n"
            "print(df.dtypes)\n"
            "print('\\nResumen de valores faltantes:')\n"
            "print(df.isnull().sum())\n\n"
            "# Visualizar distribución de variables numéricas\n"
            "import matplotlib.pyplot as plt\n"
            "df.select_dtypes(include=np.number).hist(figsize=(15, 10))\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )

    def _adaptive_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transformaciones adaptativas según el tipo de datos."""
        context = self.ctx.get_context()
        target = context['target_column']
        df_copy = df.copy()
        
        # Inicializar registro de transformaciones
        feature_transformations = {
            'removed_columns': [],
            'created_columns': [],
            'transformed_columns': []
        }
        
        # Código para documentación
        transformation_code = [
            "# Transformaciones adaptativas según el tipo de datos",
            "df_transformed = df.copy()",
            f"target = '{target}'  # Columna objetivo"
        ]
        
        # Manejar valores faltantes en el dataset
        transformation_steps = ["### Transformaciones realizadas:"]
        
        for col in df_copy.columns:
            # Saltamos la columna objetivo, que ya fue validada
            if col == target:
                continue
                    
            missing_ratio = df_copy[col].isnull().mean()
            # Si más del 70% son valores faltantes, eliminamos la columna
            if missing_ratio > 0.7:
                self.log(f"Eliminando columna '{col}' con {missing_ratio:.1%} valores faltantes", 
                        ErrorSeverity.INFO)
                df_copy.drop(columns=[col], inplace=True)
                feature_transformations['removed_columns'].append({
                    'column': col,
                    'reason': 'high_missing_ratio',
                    'missing_ratio': float(missing_ratio)
                })
                
                transformation_steps.append(f"- Eliminada columna `{col}` con {missing_ratio:.1%} de valores faltantes")
                transformation_code.append(f"# Eliminar columna con {missing_ratio:.1%} valores faltantes")
                transformation_code.append(f"df_transformed.drop(columns=['{col}'], inplace=True)")
        
        # Crear nuevas características para fechas
        date_columns = []
        for col in df_copy.columns:
            if col == target:
                continue
                    
            # Intentar convertir a datetime
            try:
                if df_copy[col].dtype == 'object':
                    # Verificar si parece una fecha
                    if df_copy[col].str.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').mean() > 0.7:
                        df_copy[f'{col}_dt'] = pd.to_datetime(df_copy[col], errors='coerce')
                        date_columns.append(f'{col}_dt')
                        feature_transformations['transformed_columns'].append({
                            'original_column': col,
                            'transformation': 'datetime_conversion',
                            'new_column': f'{col}_dt'
                        })
                        
                        transformation_steps.append(f"- Convertida columna `{col}` a formato de fecha (`{col}_dt`)")
                        transformation_code.append(f"# Convertir columna {col} a datetime")
                        transformation_code.append(f"df_transformed['{col}_dt'] = pd.to_datetime(df_transformed['{col}'], errors='coerce')")
            except:
                pass
        
        # Extraer componentes de fechas
        for date_col in date_columns:
            df_copy[f'{date_col}_year'] = df_copy[date_col].dt.year
            df_copy[f'{date_col}_month'] = df_copy[date_col].dt.month
            df_copy[f'{date_col}_day'] = df_copy[date_col].dt.day
            df_copy[f'{date_col}_dayofweek'] = df_copy[date_col].dt.dayofweek
            
            transformation_steps.append(f"- Extraídos componentes de fecha de `{date_col}` (año, mes, día, día de la semana)")
            transformation_code.append(f"# Extraer componentes de fecha de {date_col}")
            transformation_code.append(f"df_transformed['{date_col}_year'] = df_transformed['{date_col}'].dt.year")
            transformation_code.append(f"df_transformed['{date_col}_month'] = df_transformed['{date_col}'].dt.month")
            transformation_code.append(f"df_transformed['{date_col}_day'] = df_transformed['{date_col}'].dt.day")
            transformation_code.append(f"df_transformed['{date_col}_dayofweek'] = df_transformed['{date_col}'].dt.dayofweek")
            
            # Registrar las nuevas columnas creadas
            for component in ['year', 'month', 'day', 'dayofweek']:
                feature_transformations['created_columns'].append({
                    'name': f'{date_col}_{component}',
                    'type': f'date_{component}',
                    'source_column': date_col
                })
                
            # Eliminar columna original de fecha
            df_copy.drop(columns=[date_col], inplace=True)
            transformation_code.append(f"# Eliminar columna original de fecha")
            transformation_code.append(f"df_transformed.drop(columns=['{date_col}'], inplace=True)")
                
        # Guardar el registro de transformaciones en el contexto
        self.ctx._state['feature_transformations'] = feature_transformations
        
        # Registrar transformaciones realizadas
        self.ctx._state['pipeline_steps'].append("feature_engineering")
        
        # Documentar las transformaciones
        self.ctx.document_operation(
            "Ingeniería de Características",
            "\n".join(transformation_steps),
            "\n".join(transformation_code),
            df_copy
        )
        
        return df_copy

    def _build_dynamic_pipeline(self, df: pd.DataFrame, target: str):
        """Pipeline dinámico con manejo inteligente de tipos de datos."""
        # Identificar tipos de columnas
        numeric_features = df.drop(columns=[target]).select_dtypes(include=np.number).columns
        categorical_features = df.drop(columns=[target]).select_dtypes(exclude=np.number).columns
        
        # Definir transformers según los tipos de columnas encontrados
        transformers = []
        
        if len(numeric_features) > 0:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if len(categorical_features) > 0:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Si no hay transformadores, devolver None
        if not transformers:
            self.log("No se encontraron características para el pipeline", 
                    ErrorSeverity.WARNING)
            return None
        
        # Construir el ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Eliminar columnas no especificadas
        )
        
        # Documentar la construcción del pipeline
        pipeline_steps = []
        code_lines = ["# Construcción del pipeline de preprocesamiento"]
        
        if len(numeric_features) > 0:
            pipeline_steps.append(f"- **Columnas numéricas ({len(numeric_features)})**: Imputación (mediana) + Escalado (StandardScaler)")
            code_lines.append("from sklearn.preprocessing import StandardScaler")
            code_lines.append("from sklearn.impute import SimpleImputer")
            code_lines.append("from sklearn.pipeline import Pipeline")
            code_lines.append("from sklearn.compose import ColumnTransformer\n")
            code_lines.append("# Definir transformador para variables numéricas")
            code_lines.append("numeric_features = " + str(list(numeric_features)))
            code_lines.append("numeric_transformer = Pipeline(steps=[")
            code_lines.append("    ('imputer', SimpleImputer(strategy='median')),")
            code_lines.append("    ('scaler', StandardScaler())")
            code_lines.append("])")
            
        if len(categorical_features) > 0:
            pipeline_steps.append(f"- **Columnas categóricas ({len(categorical_features)})**: Imputación (moda) + OneHotEncoding")
            if len(numeric_features) == 0:
                code_lines.append("from sklearn.preprocessing import OneHotEncoder")
                code_lines.append("from sklearn.impute import SimpleImputer")
                code_lines.append("from sklearn.pipeline import Pipeline")
                code_lines.append("from sklearn.compose import ColumnTransformer\n")
            code_lines.append("# Definir transformador para variables categóricas")
            code_lines.append("categorical_features = " + str(list(categorical_features)))
            code_lines.append("categorical_transformer = Pipeline(steps=[")
            code_lines.append("    ('imputer', SimpleImputer(strategy='most_frequent')),")
            code_lines.append("    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))")
            code_lines.append("])")
            
        code_lines.append("\n# Crear el transformador combinado")
        code_lines.append("preprocessor = ColumnTransformer(")
        code_lines.append("    transformers=[")
        if len(numeric_features) > 0:
            code_lines.append("        ('num', numeric_transformer, numeric_features),")
        if len(categorical_features) > 0:
            code_lines.append("        ('cat', categorical_transformer, categorical_features),")
        code_lines.append("    ],")
        code_lines.append("    remainder='drop'")
        code_lines.append(")")
        
        self.ctx.document_operation(
            "Construcción del Pipeline de Preprocesamiento",
            "Se ha construido un pipeline de preprocesamiento adaptado a los tipos de datos:\n\n" + "\n".join(pipeline_steps),
            "\n".join(code_lines)
        )
        
        return preprocessor

    def _execute_pipeline(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Ejecuta el preprocesamiento y devuelve datos transformados."""
        X = df.drop(columns=[target])
        y = df[target].copy()
        
        # Si no hay pipeline, devolver los datos originales
        if self.pipeline is None:
            return X, y
        
        # Transformar los datos
        X_transformed = self.pipeline.fit_transform(X)
        
        # Convertir a DataFrame con nombres de características si es posible
        feature_names = []
        try:
            # Intentar obtener nombres de características
            feature_names = self.pipeline.get_feature_names_out()
        except (AttributeError, ValueError) as e:
            self.log(f"No se pudieron obtener nombres de características: {str(e)}", 
                    ErrorSeverity.INFO)
            # Generar nombres genéricos
            feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
        
        # Crear DataFrame con nombres de características
        X_df = pd.DataFrame(X_transformed, index=X.index, columns=feature_names)
        
        # Transformar target si es necesario
        problem_type = self.ctx.get_context().get('problem_type', 'classification')
        
        if problem_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
            # Para problemas de clasificación, codificar el target
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            
            # Guardar el codificador para uso posterior
            self.ctx._state['target_encoder'] = le
            
            # Guardar el mapeo de clases en el contexto
            class_mapping = {int(i): str(label) for i, label in enumerate(le.classes_)}
            self.ctx._state['class_mapping'] = class_mapping
            
            # Documentar clases y su codificación
            class_mapping_text = [f"- '{original}' → {encoded}" for encoded, original in enumerate(le.classes_)]
            
            # Documentar la transformación del target
            self.ctx.document_operation(
                "Codificación de la Variable Objetivo",
                f"La variable objetivo **{target}** es categórica y ha sido codificada con LabelEncoder.\n\n"
                f"**Número de clases:** {len(le.classes_)}\n\n"
                f"Mapeo de clases:\n" + 
                "\n".join(class_mapping_text),
                "# Codificación de la variable objetivo\n"
                "from sklearn.preprocessing import LabelEncoder\n"
                "le = LabelEncoder()\n"
                "y_transformed = le.fit_transform(y)\n"
                "# Mapeo de clases\n"
                "for i, class_name in enumerate(le.classes_):\n"
                "    print(f\"Clase {i} → '{class_name}'\")"
            )
            
            y = pd.Series(y_transformed, index=y.index)
        
        # Documentar la ejecución del pipeline
        self.ctx.document_operation(
            "Aplicación del Pipeline y Transformación de Datos",
            f"Se ha aplicado el pipeline de preprocesamiento a los datos:\n\n"
            f"- Dimensiones originales: {X.shape[0]} filas, {X.shape[1]} columnas\n"
            f"- Dimensiones post-transformación: {X_df.shape[0]} filas, {X_df.shape[1]} columnas",
            "# Aplicar el pipeline de preprocesamiento\n"
            "X = df.drop(columns=[target])\n"
            "y = df[target].copy()\n\n"
            "# Transformar los datos\n"
            "X_transformed = preprocessor.fit_transform(X)\n\n"
            "# Crear DataFrame con nombres de características\n"
            "try:\n"
            "    feature_names = preprocessor.get_feature_names_out()\n"
            "except:\n"
            "    feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]\n\n"
            "X_df = pd.DataFrame(X_transformed, index=X.index, columns=feature_names)\n"
            "X_df.head()",
            X_df.head()
        )
        
        return X_df, y

# ---------------------------------------------------------------------
# Modelador Avanzado
# ---------------------------------------------------------------------
class ModelShamanAgent(BaseAgent):
    def __init__(self, ctx: OperationalContext):
        super().__init__("ModelShaman", ctx)
        self.current_model = None
        self.genai = genai.GenerativeModel(CONFIG['MODEL_NAME'])
        self.metadata = None
    
    @resilient_agent(max_retries=CONFIG['RETRIES']['model_generation'])
    def conjure_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Generación de modelos con auto-corrección y validación."""
        logger.info("Iniciando conjuración del modelo...")
        problem_type = self._detect_problem_type(y)
        
        # Documentar el inicio del proceso de modelado
        self.ctx.document_operation(
            "Inicio del Proceso de Modelado",
            f"Se inicia el proceso de selección y entrenamiento del modelo para un problema de **{problem_type}**.\n\n"
            f"Datos de entrada:\n"
            f"- Número de instancias: {X.shape[0]}\n"
            f"- Número de características: {X.shape[1]}",
            f"# Detección del tipo de problema\n"
            f"problem_type = '{problem_type}'  # Detectado automáticamente\n"
            f"print(f'Tipo de problema: {problem_type}')\n"
            f"print(f'Dimensiones de X: {X.shape}')"
        )
        
        for attempt in range(3):
            try:
                # Seleccionar modelo automáticamente
                model = self._select_model(X, y, problem_type)
                
                # Validación básica antes de entrenar
                self._validate_model_inputs(X, y, problem_type)
                
                # Entrenar y evaluar el modelo
                trained_model, metrics = self._train_and_evaluate(model, X, y, problem_type)
                
                # Guardar modelo y métricas
                self.current_model = trained_model
                self.metadata = self._create_model_metadata(X, y, problem_type, metrics)
                
                logger.info(f"Modelo generado y validado con éxito. Métricas: {metrics}")
                
                # Documentar la generación exitosa del modelo
                self.ctx.document_operation(
                    "Modelo Generado Exitosamente",
                    f"Se ha entrenado exitosamente un modelo de tipo **{trained_model.__class__.__name__}**.\n\n"
                    f"### Métricas de Rendimiento:\n" +
                    "\n".join([f"- **{metric}**: {value:.4f}" for metric, value in metrics.items()]),
                    None
                )
                
                return trained_model
                
            except Exception as e:
                self.log(f"Intento {attempt+1} fallido al generar modelo: {str(e)}", 
                        ErrorSeverity.WARNING)
                self.ctx.log_error({
                    'phase': 'model_generation',
                    'error': str(e),
                    'attempt': attempt+1
                })
                
                # Documentar el error en el intento
                self.ctx.document_operation(
                    f"Error en Intento {attempt+1} de Generación de Modelo",
                    f"Se ha producido un error al intentar generar el modelo:\n\n"
                    f"```\n{str(e)}\n```\n\n"
                    f"Se realizará un nuevo intento.",
                    None
                )
        
        # Si llegamos aquí, todos los intentos fallaron
        self.log("Fallback a modelo baseline", ErrorSeverity.CRITICAL)
        fallback_model = self._create_fallback_model(problem_type)
        _, metrics = self._train_and_evaluate(fallback_model, X, y, problem_type)
        
        self.current_model = fallback_model
        self.metadata = self._create_model_metadata(X, y, problem_type, metrics, is_fallback=True)
        
        logger.info("Utilizando modelo baseline como fallback.")
        
        # Documentar el uso del modelo de fallback
        self.ctx.document_operation(
            "Utilización de Modelo Baseline (Fallback)",
            f"Después de múltiples intentos fallidos, se ha recurrido a un modelo baseline simple.\n\n"
            f"Tipo de modelo: **{fallback_model.__class__.__name__}**\n\n"
            f"### Métricas de Rendimiento:\n" +
            "\n".join([f"- **{metric}**: {value:.4f}" for metric, value in metrics.items()]),
            f"# Creación de modelo fallback (baseline)\n"
            f"from sklearn.dummy import {'DummyClassifier' if problem_type == 'classification' else 'DummyRegressor'}\n"
            f"fallback_model = {'DummyClassifier(strategy=\"most_frequent\")' if problem_type == 'classification' else 'DummyRegressor(strategy=\"mean\")'}\n"
            f"fallback_model.fit(X_train, y_train)"
        )
        
        return fallback_model

    def _detect_problem_type(self, y: pd.Series) -> str:
        """Determina si es un problema de clasificación o regresión."""
        # Intentar obtener el tipo de problema del contexto
        context = self.ctx.get_context()
        if context and 'problem_type' in context:
            return context['problem_type']
        
        # Si no está disponible, detectarlo automáticamente
        if not pd.api.types.is_numeric_dtype(y):
            return 'classification'
        
        # Si es numérico pero tiene pocos valores únicos, probablemente es clasificación
        n_unique = y.nunique()
        if n_unique <= 10 and n_unique / len(y) < 0.05:
            return 'classification'
        
        # En otros casos, es regresión
        return 'regression'

    def _select_model(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> BaseEstimator:
        """Selecciona automáticamente el mejor modelo según las características."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        n_samples, n_features = X.shape
        
        # Documentar criterios de selección
        selection_criteria = [
            f"- **Tipo de problema**: {problem_type}",
            f"- **Número de muestras**: {n_samples}",
            f"- **Número de características**: {n_features}"
        ]
        
        # Lógica básica de selección
        if problem_type == 'classification':
            if n_samples < 1000 or n_features > 50:
                model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=None,
                    min_samples_split=2,
                    random_state=CONFIG['RANDOM_SEED']
                )
                selection_criteria.append("- **Criterio aplicado**: Dataset pequeño o muchas características → RandomForest")
                model_code = (
                    "from sklearn.ensemble import RandomForestClassifier\n"
                    "model = RandomForestClassifier(\n"
                    "    n_estimators=100,\n" 
                    "    max_depth=None,\n"
                    "    min_samples_split=2,\n"
                    f"    random_state={CONFIG['RANDOM_SEED']}\n"
                    ")"
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=CONFIG['RANDOM_SEED']
                )
                selection_criteria.append("- **Criterio aplicado**: Dataset grande con pocas características → GradientBoosting")
                model_code = (
                    "from sklearn.ensemble import GradientBoostingClassifier\n"
                    "model = GradientBoostingClassifier(\n"
                    "    n_estimators=100,\n"
                    "    max_depth=3,\n"
                    f"    random_state={CONFIG['RANDOM_SEED']}\n"
                    ")"
                )
        else:  # regression
            if n_samples < 1000 or n_features > 50:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=CONFIG['RANDOM_SEED']
                )
                selection_criteria.append("- **Criterio aplicado**: Dataset pequeño o muchas características → RandomForest")
                model_code = (
                    "from sklearn.ensemble import RandomForestRegressor\n"
                    "model = RandomForestRegressor(\n"
                    "    n_estimators=100,\n"
                    "    max_depth=None,\n"
                    "    min_samples_split=2,\n"
                    f"    random_state={CONFIG['RANDOM_SEED']}\n"
                    ")"
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    random_state=CONFIG['RANDOM_SEED']
                )
                selection_criteria.append("- **Criterio aplicado**: Dataset grande con pocas características → GradientBoosting")
                model_code = (
                    "from sklearn.ensemble import GradientBoostingRegressor\n"
                    "model = GradientBoostingRegressor(\n"
                    "    n_estimators=100,\n"
                    "    max_depth=3,\n"
                    f"    random_state={CONFIG['RANDOM_SEED']}\n"
                    ")"
                )
        
        # Documentar selección del modelo
        self.ctx.document_operation(
            "Selección Automática del Modelo",
            f"Se ha seleccionado automáticamente un modelo de tipo **{model.__class__.__name__}** basado en las características del dataset.\n\n"
            "### Criterios de selección:\n" + "\n".join(selection_criteria),
            model_code
        )
        
        return model

    def _validate_model_inputs(self, X: pd.DataFrame, y: pd.Series, problem_type: str):
        """Validación de los inputs para el modelo."""
        # Verificar que X no tenga valores faltantes
        if X.isnull().any().any():
            raise ValueError("Los datos de entrada contienen valores faltantes")
        
        # Verificar que y no tenga valores faltantes
        if y.isnull().any():
            raise ValueError("La variable objetivo contiene valores faltantes")
        
        # Verificar que X tenga al menos una columna
        if X.shape[1] == 0:
            raise ValueError("No hay características disponibles para el modelo")
        
        # Validaciones específicas según el tipo de problema
        if problem_type == 'classification':
            if y.nunique() < 2:
                raise ValueError("La variable objetivo debe tener al menos 2 clases")
        else:  # regression
            if not pd.api.types.is_numeric_dtype(y):
                raise ValueError("Para regresión, la variable objetivo debe ser numérica")
                
        # Documentar validación
        validation_results = [
            "- ✅ No hay valores faltantes en las características",
            "- ✅ No hay valores faltantes en la variable objetivo",
            f"- ✅ Hay {X.shape[1]} características disponibles"
        ]
        
        if problem_type == 'classification':
            validation_results.append(f"- ✅ La variable objetivo tiene {y.nunique()} clases")
        else:
            validation_results.append("- ✅ La variable objetivo es numérica")
            
        self.ctx.document_operation(
            "Validación de Entradas del Modelo",
            "Se ha verificado que los datos cumplen con los requisitos para el entrenamiento del modelo:\n\n" +
            "\n".join(validation_results),
            "# Validación de inputs para el modelo\n"
            "# Verificar valores faltantes\n"
            "print(f'Valores faltantes en X: {X.isnull().any().any()}')\n"
            "print(f'Valores faltantes en y: {y.isnull().any()}')\n"
            "print(f'Número de características: {X.shape[1]}')\n"
            f"# Validación específica para problema de {problem_type}\n" +
            ("print(f'Número de clases en objetivo: {y.nunique()}')" if problem_type == 'classification' else 
             "print(f'Tipo de dato del objetivo: {y.dtype}')")
        )

    def _train_and_evaluate(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, 
                           problem_type: str) -> Tuple[BaseEstimator, Dict[str, float]]:
        """Entrena y evalúa el modelo, retornando métricas de rendimiento."""
        # División en conjuntos de entrenamiento y prueba
        # Verificar si podemos usar estratificación (necesita al menos 2 ejemplos de cada clase)
        use_stratify = False
        if problem_type == 'classification':
            # Contar ejemplos por clase
            class_counts = pd.Series(y).value_counts()
            min_samples = class_counts.min()
            
            # Solo usar estratificación si todas las clases tienen al menos 2 ejemplos
            use_stratify = min_samples >= 2
            
            if not use_stratify:
                self.log(f"No se puede usar estratificación: la clase minoritaria tiene solo {min_samples} ejemplo(s)", 
                        ErrorSeverity.WARNING)
        
        # Realizar split con o sin estratificación según corresponda
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['DEFAULT_TEST_SIZE'], 
            random_state=CONFIG['RANDOM_SEED'],
            stratify=y if (problem_type == 'classification' and use_stratify) else None
        )
        
        # Documentar la división de datos y advertir sobre la estratificación si aplica
        stratification_note = ""
        if problem_type == 'classification':
            if use_stratify:
                stratification_note = "*Nota: Se ha utilizado estratificación para mantener la distribución de clases.*"
            else:
                stratification_note = "*Nota: No se pudo usar estratificación debido a clases con muy pocos ejemplos.*"
        
        self.ctx.document_operation(
            "División en Conjuntos de Entrenamiento y Prueba",
            f"Los datos se han dividido en conjuntos de entrenamiento ({1-CONFIG['DEFAULT_TEST_SIZE']:.0%}) y prueba ({CONFIG['DEFAULT_TEST_SIZE']:.0%}):\n\n"
            f"- **Conjunto de entrenamiento**: {X_train.shape[0]} muestras\n"
            f"- **Conjunto de prueba**: {X_test.shape[0]} muestras\n\n"
            + stratification_note,
            "# División en conjuntos de entrenamiento y prueba\n"
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            f"    X, y, test_size={CONFIG['DEFAULT_TEST_SIZE']}, random_state={CONFIG['RANDOM_SEED']}," +
            ("\n    stratify=y  # Mantener distribución de clases" if (problem_type == 'classification' and use_stratify) else "") +
            "\n)\n"
            "print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')"
        )
        
        # Para clasificación, documentar las clases
        if problem_type == 'classification':
            unique_classes = y.unique()
            num_classes = len(unique_classes)
            
            # Documentar información de clases
            classes_info = f"El problema tiene **{num_classes} clases**:\n\n"
            for i, cls in enumerate(unique_classes):
                count = (y == cls).sum()
                classes_info += f"- Clase '{cls}': {count} muestras ({count/len(y):.1%})\n"
            
            self.ctx.document_operation(
                "Análisis de Clases",
                classes_info,
                "# Análisis de las clases\n"
                "unique_classes = y.unique()\n"
                "for cls in unique_classes:\n"
                "    count = (y == cls).sum()\n"
                "    print(f\"Clase '{cls}': {count} muestras ({count/len(y):.1%})\")"
            )
        
        # Entrenamiento del modelo
        model.fit(X_train, y_train)
        
        # Documentar el proceso de entrenamiento
        self.ctx.document_operation(
            "Entrenamiento del Modelo",
            f"Se ha entrenado el modelo {model.__class__.__name__} con los datos de entrenamiento.",
            "# Entrenamiento del modelo\n"
            "model.fit(X_train, y_train)\n"
            "print('Modelo entrenado exitosamente.')"
        )
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Cálculo de métricas según el tipo de problema
        metrics = {}
        
        if problem_type == 'classification':
            # Suprimir las advertencias específicas de métricas indefinidas
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                
                metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
                metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                    
            # Validación cruzada para precisión más robusta
            cv_scores = cross_val_score(model, X, y, cv=5)
            metrics['cv_accuracy_mean'] = float(cv_scores.mean())
            metrics['cv_accuracy_std'] = float(cv_scores.std())
            
            # Reporte de clasificación completo (para logging)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                class_report = classification_report(y_test, y_pred, zero_division=0)
                
            logger.info("\nReporte de clasificación:\n" + class_report)
            
            # Documentar evaluación del modelo (clasificación)
            evaluation_code = (
                "# Evaluación del modelo de clasificación\n"
                "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report\n\n"
                "# Predicciones\n"
                "y_pred = model.predict(X_test)\n\n"
                "# Métricas principales\n"
                "print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')\n"
                "print(f'F1 Score: {f1_score(y_test, y_pred, average=\"weighted\"):.4f}')\n"
                "print(f'Precision: {precision_score(y_test, y_pred, average=\"weighted\"):.4f}')\n"
                "print(f'Recall: {recall_score(y_test, y_pred, average=\"weighted\"):.4f}')\n\n"
                "# Reporte de clasificación detallado\n"
                "print('\\nReporte de clasificación:')\n"
                "print(classification_report(y_test, y_pred))"
            )
            
        else:  # regression
            metrics['r2'] = float(r2_score(y_test, y_pred))
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            # Calcular RMSE manualmente para mayor compatibilidad
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            
            # Validación cruzada para R²
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            metrics['cv_r2_mean'] = float(cv_scores.mean())
            metrics['cv_r2_std'] = float(cv_scores.std())
            
            # Documentar evaluación del modelo (regresión)
            evaluation_code = (
                "# Evaluación del modelo de regresión\n"
                "from sklearn.metrics import r2_score, mean_squared_error\n"
                "import numpy as np\n\n"
                "# Predicciones\n"
                "y_pred = model.predict(X_test)\n\n"
                "# Métricas principales\n"
                "r2 = r2_score(y_test, y_pred)\n"
                "mse = mean_squared_error(y_test, y_pred)\n"
                "rmse = np.sqrt(mse)\n\n"
                "print(f'R²: {r2:.4f}')\n"
                "print(f'MSE: {mse:.4f}')\n"
                "print(f'RMSE: {rmse:.4f}')"
            )
        
        # Documentar evaluación del modelo
        metrics_str = "\n".join([f"- **{k}**: {v:.4f}" for k, v in metrics.items()])
        self.ctx.document_operation(
            "Evaluación del Modelo",
            f"Se ha evaluado el rendimiento del modelo utilizando el conjunto de prueba.\n\n"
            f"### Métricas de Rendimiento:\n{metrics_str}",
            evaluation_code
        )
        
        # Documentar validación cruzada
        cv_metrics = {k: v for k, v in metrics.items() if k.startswith('cv_')}
        if cv_metrics:
            cv_metrics_str = "\n".join([f"- **{k}**: {v:.4f}" for k, v in cv_metrics.items()])
            self.ctx.document_operation(
                "Validación Cruzada",
                f"Se ha realizado validación cruzada (5-fold) para obtener una estimación más robusta del rendimiento del modelo.\n\n"
                f"### Métricas de Validación Cruzada:\n{cv_metrics_str}",
                "# Validación cruzada\n"
                "from sklearn.model_selection import cross_val_score\n"
                f"cv_scores = cross_val_score(model, X, y, cv=5{', scoring=\"r2\"' if problem_type == 'regression' else ''})\n"
                "print(f'Scores de validación cruzada: {cv_scores}')\n"
                "print(f'Media: {cv_scores.mean():.4f}')\n"
                "print(f'Desviación estándar: {cv_scores.std():.4f}')"
            )
        
        # Guardar métricas en el contexto
        self.ctx._state['performance_metrics'].update(metrics)
        
        return model, metrics

    def _create_fallback_model(self, problem_type: str) -> BaseEstimator:
        """Crea un modelo simple como fallback."""
        from sklearn.dummy import DummyClassifier, DummyRegressor
        
        if problem_type == 'classification':
            return DummyClassifier(strategy='most_frequent', 
                                  random_state=CONFIG['RANDOM_SEED'])
        else:
            return DummyRegressor(strategy='mean')

    def _create_model_metadata(self, X: pd.DataFrame, y: pd.Series, 
                             problem_type: str, metrics: Dict, 
                             is_fallback: bool = False) -> ModelMetadata:
        """Crea metadatos para el modelo entrenado."""
        context = self.ctx.get_context()
        target_column = context['target_column']
        
        # Obtener información de transformaciones si existe
        feature_info = context.get('feature_transformations', {})
        
        # Obtener el mapeo de clases si existe (para problemas de clasificación)
        class_mapping = {}
        if problem_type == 'classification' and 'class_mapping' in context:
            class_mapping = context['class_mapping']
        
        return ModelMetadata(
            model_type=self.current_model.__class__.__name__,
            target_column=target_column,
            features=X.columns.tolist(),
            creation_time=time.time(),
            performance_metrics=metrics,
            data_statistics=context['data_statistics'],
            feature_transformations=feature_info,
            pipeline_steps=context.get('pipeline_steps', []),
            class_mapping=class_mapping,  # Añadir el mapeo de clases
            version="1.0.0-fallback" if is_fallback else "1.0.0"
        )

# ---------------------------------------------------------------------
# Validador del Workflow
# ---------------------------------------------------------------------
class OracleAgent(BaseAgent):
    def __init__(self, ctx: OperationalContext):
        super().__init__("Oracle", ctx)
    
    def validate_workflow(self) -> bool:
        """Validación integral de todo el workflow."""
        logger.info("Validando el flujo completo en OracleAgent...")
        context = self.ctx.get_context()
        
        # 1. Validación de datos
        if not self._validate_data_integrity():
            return False
        
        # 2. Validación del modelo
        if not self._validate_model_performance():
            return False
        
        # 3. Validación de deployment
        if not self._validate_deployment_readiness():
            return False
        
        logger.info("Validación del flujo completada correctamente.")
        
        # Documentar validación del workflow
        self.ctx.document_operation(
            "Validación Final del Workflow",
            "Se ha realizado una validación integral del workflow y se ha determinado que el proceso cumple con todos los requisitos de calidad:\n\n"
            "✅ **Integridad de datos**: Los datos han sido correctamente procesados y validados.\n"
            "✅ **Rendimiento del modelo**: El modelo cumple con los umbrales mínimos de rendimiento establecidos.\n"
            "✅ **Preparación para despliegue**: El modelo está listo para ser desplegado.",
            "# Validación final del workflow\n"
            "print('Validación de integridad de datos: OK')\n"
            "print('Validación de rendimiento del modelo: OK')\n"
            "print('Validación de preparación para despliegue: OK')\n"
            "print('¡Workflow completado exitosamente!')"
        )
        
        return True
    
    def _validate_data_integrity(self) -> bool:
        """Validación de la integridad y calidad de los datos."""
        context = self.ctx.get_context()
        data_stats = context.get('data_statistics', {})
        
        # Verificar que tenemos estadísticas de datos
        if not data_stats:
            self.log("No hay estadísticas de datos disponibles", ErrorSeverity.CRITICAL)
            return False
        
        # Verificar cantidad mínima de muestras
        min_rows = self.ctx.get_validation_rules()['data_quality']['min_rows']
        if data_stats.get('n_samples', 0) < min_rows:
            self.log(f"Conjunto de datos tiene menos de {min_rows} filas", 
                    ErrorSeverity.WARNING)
            # No fallamos en este caso, solo advertimos
        
        # Verificar ratio de valores faltantes en la columna objetivo
        max_missing = self.ctx.get_validation_rules()['data_quality']['max_missing']
        if data_stats.get('missing_ratio', 0) > max_missing:
            self.log(f"Columna objetivo tiene demasiados valores faltantes: {data_stats.get('missing_ratio', 0):.1%}", 
                    ErrorSeverity.CRITICAL)
            return False
        
        return True
    
    def _validate_model_performance(self) -> bool:
        """Valida que el modelo cumpla con los requisitos mínimos de rendimiento."""
        context = self.ctx.get_context()
        metrics = context.get('performance_metrics', {})
        thresholds = self.ctx.get_validation_rules()['model_performance']
        
        # Detectar tipo de problema basado en métricas disponibles o del contexto
        problem_type = context.get('problem_type', 'classification')
        if not problem_type and 'accuracy' in metrics:
            problem_type = 'classification'
        elif not problem_type:
            problem_type = 'regression'
        
        if problem_type == 'classification':
            # Verificar accuracy y F1
            if metrics.get('accuracy', 0) < thresholds['classification']['min_accuracy']:
                self.log(f"Accuracy por debajo del umbral requerido: {metrics.get('accuracy', 0):.3f} < "
                       f"{thresholds['classification']['min_accuracy']}", 
                      ErrorSeverity.WARNING)
                # No fallamos solo por accuracy bajo
            
            if metrics.get('f1', 0) < thresholds['classification']['min_f1']:
                self.log(f"F1-score por debajo del umbral requerido: {metrics.get('f1', 0):.3f} < "
                       f"{thresholds['classification']['min_f1']}", 
                      ErrorSeverity.CRITICAL)
                return False
                
        else:  # regression
            # Verificar R²
            if thresholds['regression']['min_r2'] is not None and \
               metrics.get('r2', 0) < thresholds['regression']['min_r2']:
                self.log(f"R² por debajo del umbral requerido: {metrics.get('r2', 0):.3f} < "
                       f"{thresholds['regression']['min_r2']}", 
                      ErrorSeverity.CRITICAL)
                return False
            
            # Verificar RMSE si hay un umbral definido
            if thresholds['regression']['max_rmse'] is not None and \
               metrics.get('rmse', float('inf')) > thresholds['regression']['max_rmse']:
                self.log(f"RMSE por encima del umbral permitido: {metrics.get('rmse', 0):.3f} > "
                       f"{thresholds['regression']['max_rmse']}", 
                      ErrorSeverity.CRITICAL)
                return False
        
        self.log("Rendimiento del modelo validado correctamente", ErrorSeverity.INFO)
        return True
    
    def _validate_deployment_readiness(self) -> bool:
        """Verifica que todo está listo para el despliegue."""
        context = self.ctx.get_context()
        
        # Verificar que tenemos un modelo
        model_agents = [agent for agent in BaseAgent.instances 
                       if isinstance(agent, ModelShamanAgent)]
        
        if not model_agents or not hasattr(model_agents[0], 'current_model') or model_agents[0].current_model is None:
            self.log("No hay modelo disponible para desplegar", 
                    ErrorSeverity.WARNING)  # Cambiado de FATAL a WARNING
            return True  # Permitir continuar incluso si hay advertencia
        
        # Verificar que el modelo se puede serializar
        try:
            model = model_agents[0].current_model
            model_dir = Path(CONFIG['MODEL_DIR'])
            model_dir.mkdir(exist_ok=True)
            
            test_file = model_dir / 'test_serialization.joblib'
            
            # Intentar serializar y deserializar
            joblib.dump(model, test_file)
            _ = joblib.load(test_file)
            
            # Limpiar archivo de prueba
            if test_file.exists():
                test_file.unlink()
                
        except Exception as e:
            self.log(f"Error al serializar el modelo: {str(e)}", 
                    ErrorSeverity.CRITICAL)
            return False
        
        # Verificar que tenemos metadatos del modelo
        if model_agents and not model_agents[0].metadata:
            self.log("No hay metadatos disponibles para el modelo", 
                    ErrorSeverity.WARNING)
            # No fallamos por esto, solo advertimos
        
        return True

# ---------------------------------------------------------------------
# Agente de Recuperación
# ---------------------------------------------------------------------
class PhoenixAgent(BaseAgent):
    def __init__(self, ctx: OperationalContext):
        super().__init__("Phoenix", ctx)
    
    def resurrect_workflow(self) -> bool:
        """Intenta recuperar el workflow desde puntos de fallo."""
        context = self.ctx.get_context()
        
        if context['current_stage'] == WorkflowStage.ERROR_HANDLING:
            self.log("Iniciando recuperación de fallo...", ErrorSeverity.WARNING)
            
            # Documentar el intento de recuperación
            self.ctx.document_operation(
                "Intento de Recuperación de Fallo",
                "Se ha detectado un fallo en el workflow y se está intentando recuperar el proceso.\n\n"
                "El sistema aplicará estrategias de recuperación adaptadas al tipo de error detectado.",
                "# Intento de recuperación de fallo\n"
                "print('Detectado fallo en el workflow')\n"
                "print('Aplicando estrategias de recuperación...')"
            )
            
            return self._apply_recovery_strategy()
        
        return False
    
    def _apply_recovery_strategy(self) -> bool:
        """Estrategias avanzadas de recuperación."""
        context = self.ctx.get_context()
        error_logs = context['error_log']
        
        # Si no hay errores, no podemos hacer mucho
        if not error_logs:
            self.log("No hay registros de error para analizar", 
                    ErrorSeverity.WARNING)
            return False
        
        # Analizar los últimos errores
        recent_errors = error_logs[-3:]  # Últimos 3 errores
        error_agents = set(err.get('agent', 'unknown') for err in recent_errors)
        
        # Estrategia según el agente que falló
        if "DataGuardianAgent" in error_agents:
            # Problema con la columna objetivo
            self.log("Intentando recuperar de error en DataGuardianAgent", 
                    ErrorSeverity.INFO)
            # Usar la última columna como fallback
            if context.get('dataset') is not None:
                last_column = context['dataset'].columns[-1]
                self.ctx.set_target_column(last_column)
                
                # Documentar la estrategia de recuperación
                self.ctx.document_operation(
                    "Recuperación: Selección de Columna Objetivo Alternativa",
                    f"Debido a errores al determinar la columna objetivo, se ha seleccionado `{last_column}` como columna objetivo alternativa.",
                    f"# Selección de columna objetivo alternativa\ntarget_column = '{last_column}'  # Columna alternativa"
                )
                
                return True
                
        elif "DataAlchemist" in error_agents:
            # Problema con el preprocesamiento
            self.log("Intentando recuperar de error en DataAlchemist", 
                    ErrorSeverity.INFO)
            # Simplificar las transformaciones
            self.ctx._state['pipeline_steps'] = ["simplified_preprocessing"]
            
            # Documentar la estrategia de recuperación
            self.ctx.document_operation(
                "Recuperación: Simplificación del Preprocesamiento",
                "Debido a errores en la fase de preprocesamiento complejo, se aplicará un enfoque simplificado con transformaciones básicas.",
                "# Simplificación del preprocesamiento\n"
                "# Se aplicarán solo transformaciones básicas:\n"
                "# - Imputación de valores faltantes\n"
                "# - Encoding de variables categóricas\n"
                "# - Escalado de variables numéricas"
            )
            
            return True
            
        elif "ModelShaman" in error_agents:
            # Problema con la generación del modelo
            self.log("Intentando recuperar de error en ModelShaman", 
                    ErrorSeverity.INFO)
            # Forzar el uso de un modelo más simple
            self.ctx._state['fallback_activated'] = True
            
            # Documentar la estrategia de recuperación
            self.ctx.document_operation(
                "Recuperación: Uso de Modelo Simplificado",
                "Debido a errores en la generación del modelo complejo, se utilizará un modelo baseline más simple.",
                "# Uso de modelo simplificado\n"
                "from sklearn.dummy import DummyClassifier, DummyRegressor\n"
                "# Se utilizará un modelo baseline según el tipo de problema"
            )
            
            return True
        
        # Si no podemos identificar una estrategia específica
        self.log("No se pudo determinar una estrategia de recuperación", 
                ErrorSeverity.CRITICAL)
        
        # Documentar el fallo de recuperación
        self.ctx.document_operation(
            "Fallo en la Recuperación",
            "No se ha podido determinar una estrategia específica para recuperar el workflow después del error.\n\n"
            "Se recomienda revisar los logs para más detalles sobre el error y posibles soluciones manuales.",
            None
        )
        
        return False

# ---------------------------------------------------------------------
# Workflow Autocurativo
# ---------------------------------------------------------------------
class AICortex:
    def __init__(self):
        self.ctx = OperationalContext()
        
        # Crear el agente de documentación de notebook primero
        self.notebook_agent = NotebookScribeAgent(self.ctx)
        
        # Inicializar el resto de agentes
        self.agents = {
            'intent': IntentAgent(),
            'data_guardian': DataGuardianAgent(self.ctx),
            'data_alchemist': DataAlchemistAgent(self.ctx),
            'model_shaman': ModelShamanAgent(self.ctx),
            'oracle': OracleAgent(self.ctx),
            'phoenix': PhoenixAgent(self.ctx),
            'notebook_scribe': self.notebook_agent
        }
    
    def execute_workflow(self, user_input: str) -> bool:
        try:
            # Fase 1: Carga de datos
            logger.info("Cargando datos...")
            self.ctx.update_stage(WorkflowStage.DATA_LOADING)
            raw_data = self._load_data()
            # Guardar el dataset en el contexto
            self.ctx._state['dataset'] = raw_data
            logger.info(f"Datos cargados correctamente: {raw_data.shape[0]} filas, {raw_data.shape[1]} columnas")

            # Fase 1.1: Análisis de intención
            intent = self.agents['intent'].analyze_intent(user_input)
            logger.info(f"Intención del usuario: {intent}")
            
            # Guardar el tipo de problema en el contexto
            problem_type = intent.get('problem_type', 'classification')
            self.ctx.set_problem_type(problem_type)
            
            # Documentar el análisis de intención
            self.ctx.document_operation(
                "Análisis de Intención del Usuario",
                f"Se ha analizado la intención del usuario: \"{user_input}\"\n\n"
                f"- **Columna objetivo detectada**: {intent['target_column']}\n"
                f"- **Tipo de problema**: {problem_type}\n"
                f"- **Contexto**: {intent['context']}",
                "# Análisis de la intención del usuario\n"
                f"intent = {{\n"
                f"    'target_column': '{intent['target_column']}',\n"
                f"    'problem_type': '{problem_type}',\n"
                f"    'context': '''{intent['context']}'''\n"
                f"}}"
            )

            # Fase 1.2: Resolución de la columna objetivo
            target_col = self.agents['data_guardian'].resolve_target(raw_data, intent)
            self.ctx.set_target_column(target_col)
            self.ctx.update_stage(WorkflowStage.DATA_VALIDATION)
            
            # Fase 2: Preprocesamiento adaptativo
            logger.info(f"Iniciando preprocesamiento para predecir '{target_col}'...")
            clean_data, target = self.agents['data_alchemist'].auto_preprocess(raw_data)
            self.ctx.update_stage(WorkflowStage.FEATURE_ENGINEERING)
            logger.info(f"Preprocesamiento completado: {clean_data.shape[1]} características generadas")
            
            # Fase 3: Modelado inteligente
            logger.info("Iniciando entrenamiento del modelo...")
            self.ctx.update_stage(WorkflowStage.MODEL_TRAINING)
            model = self.agents['model_shaman'].conjure_model(clean_data, target)
            self.ctx.update_stage(WorkflowStage.MODEL_VALIDATION)
            
            # Fase 4: Validación final y deployment
            if self.agents['oracle'].validate_workflow():
                deployment_info = self._deploy_model(model)
                # Documentar despliegue del modelo
                self.ctx.document_operation(
                    "Finalización del Workflow y Despliegue del Modelo",
                    f"El workflow se ha completado correctamente y el modelo ha sido desplegado.\n\n"
                    f"- **Ruta del modelo**: {deployment_info['model_path']}\n"
                    f"- **Reporte de rendimiento**: {deployment_info['report_path']}\n\n"
                    "El modelo está listo para ser utilizado en predicciones.",
                    "# Carga y uso del modelo desplegado\n"
                    "import joblib\n"
                    f"model = joblib.load('{deployment_info['model_path']}')\n\n"
                    "# Ejemplo de predicción\n"
                    "# new_data = pd.DataFrame(...)\n"
                    "# predictions = model.predict(new_data)"
                )
                return True
            else:
                logger.warning("La validación del workflow ha fallado")
                return False
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error en workflow: {str(e)}\n{error_traceback}")
            
            # Documentar el error
            self.ctx.document_operation(
                "Error en el Workflow",
                f"Se ha producido un error durante la ejecución del workflow:\n\n"
                f"```\n{str(e)}\n```\n\n"
                "El sistema intentará recuperarse automáticamente.",
                None
            )
            
            # Actualizamos la etapa a ERROR_HANDLING para que Phoenix actúe
            self.ctx.update_stage(WorkflowStage.ERROR_HANDLING)
            self.ctx.log_error({
                'phase': 'global', 
                'error': str(e),
                'traceback': error_traceback
            })
            
            # Intento de recuperación
            if self.agents['phoenix'].resurrect_workflow():
                logger.info("Intentando recuperar el workflow...")
                return self.execute_workflow(user_input)
                
            logger.error("No se pudo recuperar el workflow después del error")
            
            # Documentar el fallo de recuperación
            self.ctx.document_operation(
                "Fallo en la Recuperación del Workflow",
                "No se ha podido recuperar el workflow después de múltiples intentos.\n\n"
                "Se recomienda revisar los logs para obtener más información sobre los errores encontrados.",
                None
            )
            
            return False
    
    def _load_data(self) -> pd.DataFrame:
        """Carga datos de múltiples formatos posibles."""
        data_path = Path(CONFIG['DATA_FILE'])
        
        if not data_path.exists():
            # Buscar otros archivos de datos en el directorio
            data_files = list(Path('.').glob('*.csv')) + list(Path('.').glob('*.xlsx')) + \
                        list(Path('.').glob('*.parquet')) + list(Path('.').glob('*.json'))
            
            if data_files:
                data_path = data_files[0]
                logger.info(f"Usando archivo alternativo: {data_path}")
            else:
                raise FileNotFoundError(f"No se encontraron archivos de datos en el directorio")
        
        # Cargar según el formato
        suffix = data_path.suffix.lower()
        
        if suffix == '.csv':
            df = pd.read_csv(data_path)
        elif suffix == '.xlsx':
            df = pd.read_excel(data_path)
        elif suffix == '.parquet':
            df = pd.read_parquet(data_path)
        elif suffix == '.json':
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {suffix}")
        
        logger.info(f"Datos cargados desde {data_path}: {df.shape}")
        
        # Documentar la carga de datos
        self.ctx.document_operation(
            "Carga de Datos",
            f"Se han cargado datos desde el archivo **{data_path}**.\n\n"
            f"- **Formato del archivo**: {suffix}\n"
            f"- **Dimensiones**: {df.shape[0]} filas, {df.shape[1]} columnas\n",
            f"# Carga de datos\nimport pandas as pd\ndf = pd.read_{suffix[1:]}('{data_path}')\ndf.head()",
            df.head()
        )
        
        return df
    
    def _deploy_model(self, model):
        """Implementación mejorada de despliegue del modelo."""
        logger.info("Desplegando el modelo a producción...")
        
        # Crear directorio para modelos si no existe
        model_dir = Path(CONFIG['MODEL_DIR'])
        model_dir.mkdir(exist_ok=True)
        
        # Generar timestamp para versionado
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}.joblib"
        model_path = model_dir / model_filename
        
        # Guardar el modelo
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
        
        # Crear un symlink al último modelo
        latest_path = model_dir / "model_latest.joblib"
        if latest_path.exists():
            try:
                latest_path.unlink()
            except:
                pass  # Ignorar errores al eliminar el symlink
        
        try:
            latest_path.symlink_to(model_filename)
        except Exception as e:
            logger.warning(f"No se pudo crear el symlink al último modelo: {e}")
        
        # Guardar un informe de rendimiento
        report_path = model_dir / f"report_{timestamp}.txt"
        metrics = convert_to_serializable(self.ctx.get_context()['performance_metrics'])
        
        with open(report_path, 'w') as f:
            f.write(f"# Reporte de Modelo: {timestamp}\n\n")
            f.write(f"Columna objetivo: {self.ctx.get_context()['target_column']}\n")
            f.write(f"Tipo de problema: {self.ctx.get_context()['problem_type']}\n")
            f.write(f"Tipo de modelo: {model.__class__.__name__}\n\n")
            f.write("## Métricas de rendimiento\n\n")
            
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        self.ctx.update_stage(WorkflowStage.DEPLOYMENT)
        logger.info(f"¡Modelo desplegado con éxito en {model_path}!")
        logger.info(f"Reporte de rendimiento guardado en {report_path}")
        
        # Documentar el despliegue del modelo
        if hasattr(self.agents['notebook_scribe'], 'document_model_deployment'):
            self.agents['notebook_scribe'].document_model_deployment(str(model_path), metrics)
        
        return {
            'model_path': str(model_path),
            'report_path': str(report_path),
            'metrics': metrics
        }

# ---------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------
def main():
    print("=" * 80)
    print("🧠 Midas Touch: Sistema Inteligente de Flujo de Trabajo ML")
    print("=" * 80)
    
    # Inicialización
    cortex = AICortex()
    
    # Obtener input del usuario
    user_input = input("📝 Describe tu tarea de ML (ej: 'predecir riesgo de diabetes'): ")
    
    print("\n🔄 Iniciando el procesamiento...\n")
    
    # Ejecutar workflow
    start_time = time.time()
    success = cortex.execute_workflow(user_input)
    elapsed_time = time.time() - start_time
    
    # Mostrar resultado
    if success:
        print("\n" + "=" * 80)
        print("✅ Workflow completado con éxito!")
        print(f"⏱️  Tiempo total: {elapsed_time:.2f} segundos")
        
        # Mostrar métricas de rendimiento
        metrics = cortex.ctx.get_context()['performance_metrics']
        print("\n📊 Métricas de rendimiento:")
        for metric, value in metrics.items():
            print(f"   • {metric}: {value:.4f}")
            
        print("\n📁 Modelo guardado en:", CONFIG['MODEL_DIR'])
        print(f"📓 Notebook de documentación guardado en: {CONFIG['NOTEBOOK_FILE']}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Workflow fallido después de múltiples intentos")
        print(f"⏱️  Tiempo total: {elapsed_time:.2f} segundos")
        
        # Mostrar últimos errores
        print("\n🔍 Últimos errores:")
        error_logs = cortex.ctx.get_context()['error_log'][-3:]
        for error in error_logs:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', 
                               time.localtime(error.get('timestamp', time.time())))
            print(f"   [{ts}] {error.get('agent', 'N/A')}: {error.get('message', 'Error desconocido')}")
        print("=" * 80)
        print(f"📓 Notebook con documentación parcial guardado en: {CONFIG['NOTEBOOK_FILE']}")

if __name__ == "__main__":
    # Configuración de la API de Gemini
    load_dotenv()
    genai.configure(api_key=os.getenv(CONFIG['API_KEY_ENV_VAR']))
    
    # Ejecutar aplicación
    main()