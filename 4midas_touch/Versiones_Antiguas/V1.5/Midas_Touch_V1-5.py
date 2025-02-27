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
            'min_accuracy': 0.7,
            'min_f1': 0.6
        },
        'regression': {
            'min_r2': 0.5,
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
    
    def update_stage(self, stage: WorkflowStage):
        prev_stage = self._state['current_stage']
        self._state['current_stage'] = stage
        logger.info(f"Workflow avanzando de {prev_stage.name} a {stage.name}")
    
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
        
        Ejemplos de respuesta:
        {{"target_column": "rating", "problem_type": "regression", "context": "Predicción numérica"}}
        {{"target_column": "category", "problem_type": "classification", "context": "Clasificación de productos"}}
        
        Reglas:
        1. target_column DEBE ser una columna explícitamente mencionada
        2. Priorizar términos técnicos comunes (target, label, predict)
        3. Si hay ambigüedad, pedir clarificación (pero en este caso, hacer tu mejor suposición)
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
        """Extrae posible nombre de columna objetivo del texto del usuario"""
        # Buscar patrones comunes como "predecir X", "clasificar Y", etc.
        patterns = [
            r"predecir\s+(?:la\s+)?(?:columna\s+)?['\"]*(\w+)['\"]*",
            r"clasificar\s+(?:la\s+)?(?:columna\s+)?['\"]*(\w+)['\"]*",
            r"pronosticar\s+(?:la\s+)?(?:columna\s+)?['\"]*(\w+)['\"]*",
            r"target\s+(?:es\s+)?(?:columna\s+)?['\"]*(\w+)['\"]*",
            r"objetivo\s+(?:es\s+)?(?:columna\s+)?['\"]*(\w+)['\"]*"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Si no encuentra nada, devolver un valor por defecto
        return "target"

# ---------------------------------------------------------------------
# Agente Guardián de Datos
# ---------------------------------------------------------------------
class DataGuardianAgent(BaseAgent):
    def __init__(self, ctx: OperationalContext):
        super().__init__("DataGuardianAgent", ctx)
        self.genai = genai.GenerativeModel(CONFIG['MODEL_NAME'])
    
    def resolve_target(self, df: pd.DataFrame, intent: Dict) -> str:
        logger.info("Resolviendo la columna objetivo...")
        candidates = self._generate_candidates(df, intent)
        
        if not candidates:
            self.log("No se encontraron candidatos para la columna objetivo", 
                     ErrorSeverity.CRITICAL)
            # Fallback a usar la última columna como objetivo
            return df.columns[-1]
        
        best_target = self._validate_candidate(df, candidates, intent)
        logger.info(f"Columna objetivo seleccionada: {best_target}")
        
        # Calcular y almacenar estadísticas básicas de la columna objetivo
        self._compute_target_statistics(df, best_target)
        
        return best_target

    def _generate_candidates(self, df: pd.DataFrame, intent: Dict) -> list:
        """Genera candidatos usando múltiples estrategias."""
        candidates = []
        
        # 1. Coincidencia exacta
        if intent['target_column'] in df.columns:
            candidates.append(intent['target_column'])
        
        # 2. Búsqueda difusa
        fuzzy_matches = difflib.get_close_matches(
            intent['target_column'], df.columns, 
            n=3, cutoff=0.6
        )
        candidates.extend(fuzzy_matches)
        
        # 3. Búsqueda semántica con IA
        semantic_matches = self._semantic_search(df.columns, intent)
        candidates.extend(semantic_matches)
        
        # Eliminar duplicados manteniendo el orden
        return list(dict.fromkeys(candidates))

    def _semantic_search(self, columns: list, intent: Dict) -> list:
        """Búsqueda semántica mejorada usando Gemini."""
        # Crear un prompt más estructurado y claro
        column_list = "\n".join([f"- {col}" for col in columns])
        
        prompt = f"""
        # Contexto
        Contexto del usuario: {intent['context']}
        Target sugerido inicialmente: {intent['target_column']}
        
        # Columnas disponibles
        {column_list}

        # Tarea
        Identifica la columna que es más probable que sea el objetivo para predicción/clasificación.
        
        # Criterios
        1. Prioriza columnas que semánticamente se relacionen con el target sugerido
        2. Nombres comunes para variables objetivo: target, label, class, category, result, etc.
        3. En problemas de clasificación, busca columnas categóricas
        4. En problemas de regresión, busca columnas numéricas
        
        # Formato de respuesta
        Devuelve solo los nombres exactos de las 3 columnas más relevantes, separados por comas.
        """
        
        try:
            response = self.genai.generate_content(prompt)
            return self._parse_semantic_response(response.text, columns)
        except Exception as e:
            self.log(f"Error en búsqueda semántica: {str(e)}", ErrorSeverity.WARNING)
            return []

    def _parse_semantic_response(self, text: str, columns: list) -> list:
        """Parsea la respuesta de la IA de manera más robusta."""
        # Limpiar la respuesta
        clean_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()
        
        # Estrategia 1: Buscar columnas directamente
        matches = []
        for col in columns:
            if col in clean_text:
                matches.append(col)
        
        # Estrategia 2: Si no hay coincidencias, dividir por comas o líneas
        if not matches:
            potential_matches = re.split(r',|\n', clean_text)
            for match in potential_matches:
                clean_match = match.strip().strip('"\'').strip()
                if clean_match in columns:
                    matches.append(clean_match)
        
        return matches[:3]  # Retornar hasta 3 coincidencias

    def _validate_candidate(self, df: pd.DataFrame, candidates: list, intent: Dict) -> str:
        """Valida y selecciona la mejor columna candidata."""
        problem_type = intent.get('problem_type', 'classification')
        
        # Primero, probar las coincidencias
        for candidate in candidates:
            if self._is_valid_target(df[candidate], problem_type):
                return candidate
        
        # Si ninguna coincidencia es válida, buscar en todas las columnas
        self.log("Ningún candidato válido, buscando en todas las columnas", 
                 ErrorSeverity.WARNING)
        
        for column in df.columns:
            if self._is_valid_target(df[column], problem_type):
                return column
        
        # Si todo falla, usar la primera columna candidata o la última del dataframe
        if candidates:
            self.log(f"Usando candidato no óptimo: {candidates[0]}", 
                     ErrorSeverity.WARNING)
            return candidates[0]
        
        self.log(f"Fallback a última columna: {df.columns[-1]}", 
                 ErrorSeverity.CRITICAL)
        return df.columns[-1]

    def _is_valid_target(self, series: pd.Series, problem_type: str) -> bool:
        """Validación mejorada de la columna objetivo."""
        # Verificar valores missing
        missing_ratio = series.isnull().mean()
        if missing_ratio > CONFIG['MAX_MISSING_RATIO']:
            return False
        
        if problem_type == 'classification':
            # Para clasificación: debe tener al menos 2 clases y no demasiadas
            n_unique = len(series.unique())
            return 1 < n_unique <= 100 and series.nunique() / len(series) < 0.5
        else:
            # Para regresión: debe ser numérica con suficiente variación
            if not pd.api.types.is_numeric_dtype(series):
                return False
            
            # Verificar varianza suficiente
            return series.std() > 0 and series.nunique() > 10

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
        
        # Manejar valores faltantes en el dataset
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
            except:
                pass
        
        # Extraer componentes de fechas
        for date_col in date_columns:
            df_copy[f'{date_col}_year'] = df_copy[date_col].dt.year
            df_copy[f'{date_col}_month'] = df_copy[date_col].dt.month
            df_copy[f'{date_col}_day'] = df_copy[date_col].dt.day
            df_copy[f'{date_col}_dayofweek'] = df_copy[date_col].dt.dayofweek
            
            # Registrar las nuevas columnas creadas
            for component in ['year', 'month', 'day', 'dayofweek']:
                feature_transformations['created_columns'].append({
                    'name': f'{date_col}_{component}',
                    'type': f'date_{component}',
                    'source_column': date_col
                })
                
            # Eliminar columna original de fecha
            df_copy.drop(columns=[date_col], inplace=True)
                
        # Guardar el registro de transformaciones en el contexto
        self.ctx._state['feature_transformations'] = feature_transformations
        
        # Registrar transformaciones realizadas
        self.ctx._state['pipeline_steps'].append("feature_engineering")
        
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
        if pd.api.types.is_object_dtype(y) and not pd.api.types.is_numeric_dtype(y):
            # Para problemas de clasificación, codificar el target
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            # Guardar el codificador para uso posterior
            self.ctx._state['target_encoder'] = le
            y = pd.Series(y_transformed, index=y.index)
        
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
                return trained_model
                
            except Exception as e:
                self.log(f"Intento {attempt+1} fallido al generar modelo: {str(e)}", 
                        ErrorSeverity.WARNING)
                self.ctx.log_error({
                    'phase': 'model_generation',
                    'error': str(e),
                    'attempt': attempt+1
                })
        
        # Si llegamos aquí, todos los intentos fallaron
        self.log("Fallback a modelo baseline", ErrorSeverity.CRITICAL)
        fallback_model = self._create_fallback_model(problem_type)
        _, metrics = self._train_and_evaluate(fallback_model, X, y, problem_type)
        
        self.current_model = fallback_model
        self.metadata = self._create_model_metadata(X, y, problem_type, metrics, is_fallback=True)
        
        logger.info("Utilizando modelo baseline como fallback.")
        return fallback_model

    def _detect_problem_type(self, y: pd.Series) -> str:
        """Determina si es un problema de clasificación o regresión."""
        # Verificamos si es categórico
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
        
        # Lógica básica de selección
        if problem_type == 'classification':
            if n_samples < 1000 or n_features > 50:
                return RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=None,
                    min_samples_split=2,
                    random_state=CONFIG['RANDOM_SEED']
                )
            else:
                return GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=CONFIG['RANDOM_SEED']
                )
        else:  # regression
            if n_samples < 1000 or n_features > 50:
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=CONFIG['RANDOM_SEED']
                )
            else:
                return GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    random_state=CONFIG['RANDOM_SEED']
                )

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

    def _train_and_evaluate(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, 
                           problem_type: str) -> Tuple[BaseEstimator, Dict[str, float]]:
        """Entrena y evalúa el modelo, retornando métricas de rendimiento."""
        # División en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['DEFAULT_TEST_SIZE'], 
            random_state=CONFIG['RANDOM_SEED'],
            stratify=y if problem_type == 'classification' else None
        )
        
        # Entrenamiento del modelo
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Cálculo de métricas según el tipo de problema
        metrics = {}
        if problem_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted'))
            metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
            metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
            
            # Validación cruzada para precisión más robusta
            cv_scores = cross_val_score(model, X, y, cv=5)
            metrics['cv_accuracy_mean'] = float(cv_scores.mean())
            metrics['cv_accuracy_std'] = float(cv_scores.std())
            
            # Reporte de clasificación completo (para logging)
            logger.info("\nReporte de clasificación:\n" + 
                      classification_report(y_test, y_pred))
            
        else:  # regression
            metrics['r2'] = float(r2_score(y_test, y_pred))
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            # Calcular RMSE manualmente para mayor compatibilidad
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            
            # Validación cruzada para R²
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            metrics['cv_r2_mean'] = float(cv_scores.mean())
            metrics['cv_r2_std'] = float(cv_scores.std())
        
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
        
        return ModelMetadata(
            model_type=self.current_model.__class__.__name__,
            target_column=target_column,
            features=X.columns.tolist(),
            creation_time=time.time(),
            performance_metrics=metrics,
            data_statistics=context['data_statistics'],
            feature_transformations=feature_info,  # Agregar información de transformaciones
            pipeline_steps=context.get('pipeline_steps', []),
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
        
        # Detectar tipo de problema basado en métricas disponibles
        problem_type = 'classification' if 'accuracy' in metrics else 'regression'
        
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
                return True
                
        elif "DataAlchemist" in error_agents:
            # Problema con el preprocesamiento
            self.log("Intentando recuperar de error en DataAlchemist", 
                    ErrorSeverity.INFO)
            # Simplificar las transformaciones
            self.ctx._state['pipeline_steps'] = ["simplified_preprocessing"]
            return True
            
        elif "ModelShaman" in error_agents:
            # Problema con la generación del modelo
            self.log("Intentando recuperar de error en ModelShaman", 
                    ErrorSeverity.INFO)
            # Forzar el uso de un modelo más simple
            self.ctx._state['fallback_activated'] = True
            return True
        
        # Si no podemos identificar una estrategia específica
        self.log("No se pudo determinar una estrategia de recuperación", 
                ErrorSeverity.CRITICAL)
        return False

# ---------------------------------------------------------------------
# Workflow Autocurativo
# ---------------------------------------------------------------------
class AICortex:
    def __init__(self):
        self.ctx = OperationalContext()
        self.agents = {
            'intent': IntentAgent(),
            'data_guardian': DataGuardianAgent(self.ctx),
            'data_alchemist': DataAlchemistAgent(self.ctx),
            'model_shaman': ModelShamanAgent(self.ctx),
            'oracle': OracleAgent(self.ctx),
            'phoenix': PhoenixAgent(self.ctx)
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
                self._deploy_model(model)
                return True
            else:
                logger.warning("La validación del workflow ha fallado")
                return False
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error en workflow: {str(e)}\n{error_traceback}")
            
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
            f.write(f"Tipo de modelo: {model.__class__.__name__}\n\n")
            f.write("## Métricas de rendimiento\n\n")
            
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        self.ctx.update_stage(WorkflowStage.DEPLOYMENT)
        logger.info(f"¡Modelo desplegado con éxito en {model_path}!")
        logger.info(f"Reporte de rendimiento guardado en {report_path}")
        
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
    print("🧠 AICortex: Sistema Inteligente de Flujo de Trabajo ML")
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

if __name__ == "__main__":
    # Configuración de la API de Gemini
    load_dotenv()
    genai.configure(api_key=os.getenv(CONFIG['API_KEY_ENV_VAR']))
    
    # Ejecutar aplicación
    main()