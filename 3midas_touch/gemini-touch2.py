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
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from functools import wraps
from enum import Enum, auto

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder  # Importación agregada

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("google_key"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_workflow.log"),
        logging.StreamHandler()
    ]
)

class WorkflowStage(Enum):
    DATA_LOADING = auto()
    DATA_VALIDATION = auto()
    FEATURE_ENGINEERING = auto()
    MODEL_TRAINING = auto()
    MODEL_VALIDATION = auto()
    DEPLOYMENT = auto()
    ERROR_HANDLING = auto()

class ErrorSeverity(Enum):
    WARNING = auto()
    CRITICAL = auto()
    FATAL = auto()

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
            'fallback_activated': False
        }
        self._validation_rules = {
            'data_quality': {
                'min_rows': 100,
                'max_missing': 0.3,
                'feature_variance': 0.01
            },
            'model_performance': {
                'classification': {'min_f1': 0.6},
                'regression': {'max_rmse': None}
            }
        }
    
    def update_stage(self, stage: WorkflowStage):
        self._state['current_stage'] = stage
    
    def log_error(self, error: Dict):
        # Aseguramos que el log tenga las claves mínimas para evitar errores posteriores.
        if 'timestamp' not in error:
            error['timestamp'] = time.time()
        if 'agent' not in error:
            error['agent'] = 'N/A'
        if 'message' not in error:
            error['message'] = error.get('error', 'No message provided')
        self._state['error_log'].append(error)
    
    def get_context(self) -> Dict:
        return self._state.copy()
    
    def get_validation_rules(self) -> Dict:
        return self._validation_rules

# ---------------------------------------------------------------------
# Decoradores Avanzados
# ---------------------------------------------------------------------
def resilient_agent(max_retries=3, backoff_factor=2):
    """
    Decorador para reintentar la ejecución de un método de agente
    ante excepciones, con backoff exponencial.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        self.log(f"Critical failure in {func.__name__}: {str(e)}", level=ErrorSeverity.FATAL)
                        raise
                    
                    delay = backoff_factor ** attempt
                    self.log(f"Retry {attempt+1} for {func.__name__} in {delay}s. Error: {str(e)}", 
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
# Agentes Especializados
# ---------------------------------------------------------------------
class MetaAgent(type):
    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        new_class.instances = []
        return new_class

class BaseAgent(metaclass=MetaAgent):
    def __init__(self, name: str, ctx: OperationalContext):
        self.name = name
        self.ctx = ctx
        self.logger = self._configure_logger()
    
    def _configure_logger(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def log(self, message: str, level: ErrorSeverity = ErrorSeverity.WARNING):
        log_entry = {
            'timestamp': time.time(),
            'agent': self.name,
            'message': message,
            'level': level
        }
        self.ctx.log_error(log_entry)
        getattr(self.logger, level.name.lower())(message)

class IntentAgent:
    """
    Agente para analizar la intención del usuario:
    - Extrae 'target_column'
    - Determina 'problem_type'
    - Devuelve 'context' relevante
    """
    def __init__(self):
        self.logger = logging.getLogger("IntentAgent")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.genai = genai.GenerativeModel('gemini-pro')
    
    def analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """Analiza la intención del usuario y extrae parámetros clave"""
        print("Analizando la intención del usuario...")
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
        
        for _ in range(3):  # Reintentos simples
            try:
                response = self.genai.generate_content(prompt)
                parsed = self._safe_parse(response.text)
                self.logger.info(f"Intención detectada: {parsed}")
                return parsed
            except Exception as e:
                self.logger.error(f"Error analizando intención: {str(e)}")
                time.sleep(2)
        
        raise RuntimeError("No se pudo determinar la intención del usuario")

    def _safe_parse(self, text: str) -> Dict:
        """Parseo seguro de la respuesta de la IA (JSON)"""
        try:
            clean_text = re.sub(r'```json|```', '', text).strip()
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
            "problem_type": (problem.group(1).lower() if problem else None),
            "context": text
        }

class DataGuardianAgent(BaseAgent):
    """
    Agente encargado de identificar/validar la columna objetivo.
    Integra coincidencia exacta, fuzzy matching y búsqueda semántica.
    """
    def __init__(self, ctx: OperationalContext):
        super().__init__("DataGuardianAgent", ctx)
        self.genai = genai.GenerativeModel('gemini-pro')
    
    def resolve_target(self, df: pd.DataFrame, intent: Dict) -> str:
        """Resolución robusta de la columna objetivo."""
        print("Resolviendo la columna objetivo...")
        candidates = self._generate_candidates(df, intent)
        best_target = self._validate_candidate(df, candidates, intent)
        print(f"Columna objetivo seleccionada: {best_target}")
        return best_target

    def _generate_candidates(self, df: pd.DataFrame, intent: Dict) -> list:
        """Genera candidatos usando múltiples estrategias."""
        # 1. Coincidencia exacta
        if intent['target_column'] in df.columns:
            return [intent['target_column']]
        
        # 2. Búsqueda difusa
        fuzzy_matches = difflib.get_close_matches(
            intent['target_column'], df.columns, 
            n=3, cutoff=0.6
        )
        
        # 3. Búsqueda semántica con IA
        semantic_matches = self._semantic_search(df.columns, intent)
        
        return list(set(fuzzy_matches + semantic_matches))

    def _semantic_search(self, columns: list, intent: Dict) -> list:
        """Búsqueda semántica usando Gemini."""
        prompt = f"""
        Contexto completo del usuario: {intent['context']}
        Target sugerido inicialmente: {intent['target_column']}
        Columnas disponibles: {columns}

        Tarea:
        1. Identificar la columna objetivo REAL basada en el contexto
        2. Considerar sinónimos y relaciones semánticas
        3. Priorizar columnas mencionadas explícitamente

        Devuelve solo las 3 columnas más relevantes en orden de prioridad.
        """
        response = self.genai.generate_content(prompt)
        return self._parse_semantic_response(response.text, columns)

    def _parse_semantic_response(self, text: str, columns: list) -> list:
        """Parsea la respuesta de la IA."""
        matches = []
        for line in text.split('\n'):
            clean_line = re.sub(r'\d+\.\s*', '', line).strip()
            if clean_line in columns:
                matches.append(clean_line)
        return matches[:3]

    def _validate_candidate(self, df: pd.DataFrame, candidates: list, intent: Dict) -> str:
        """Valida la mejor candidata según el tipo de problema."""
        for candidate in candidates:
            if self._is_valid_target(df[candidate], intent['problem_type']):
                return candidate
        
        raise ValueError(f"No se encontró una columna objetivo válida. Candidatos: {candidates}")

    def _is_valid_target(self, series: pd.Series, problem_type: str) -> bool:
        """Valida si la serie es un target adecuado."""
        if problem_type == 'classification':
            # Debe haber más de una clase y un número razonable de categorías
            return 1 < len(series.unique()) <= 100
        else:
            # En regresión, el target debe ser numérico
            return pd.api.types.is_numeric_dtype(series)

class DataAlchemistAgent(BaseAgent):
    """
    Agente encargado de:
    - Validar dataset de forma avanzada
    - Ingeniería de características
    - Construcción dinámica de pipelines
    - Preprocesamiento final
    """
    def __init__(self, ctx: OperationalContext):
        super().__init__("DataAlchemist", ctx)
        self.pipeline = None
    
    @resilient_agent(max_retries=3)
    def auto_preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesamiento adaptativo con auto-detección de estrategias."""
        print("Iniciando preprocesamiento de datos...")
        context = self.ctx.get_context()
        target = context['target_column']
        
        try:
            # 1. Validación inicial
            self._validate_dataset(df, target)
            
            # 2. Ingeniería de características adaptativa
            df_transformed = self._adaptive_feature_engineering(df)
            
            # 3. Construcción dinámica del pipeline
            self.pipeline = self._build_dynamic_pipeline(df_transformed, target)
            
            # 4. Ejecución con validación
            X, y = self._execute_pipeline(df_transformed, target)
            print("Preprocesamiento completado correctamente.")
            return X, y
        
        except Exception as e:
            self.log(f"Error en preprocesamiento: {str(e)}", ErrorSeverity.CRITICAL)
            raise

    def _validate_dataset(self, df: pd.DataFrame, target: str):
        """Validación avanzada de integridad de datos."""
        if target not in df.columns:
            raise ValueError(f"La columna objetivo '{target}' no existe en el DataFrame.")
        
        if df[target].isnull().mean() > 0.4:
            self.log(f"Columna objetivo '{target}' tiene >40% valores faltantes", ErrorSeverity.CRITICAL)
            raise ValueError("Problemas graves en columna objetivo")
        
        if len(df[target].unique()) == 1:
            self.log(f"Columna objetivo '{target}' tiene un único valor", ErrorSeverity.FATAL)
            raise ValueError("Target constante")

    def _adaptive_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transformaciones adaptativas basadas en análisis de datos (ejemplo simplificado)."""
        # Aquí se pueden implementar transformaciones complejas según el análisis
        return df

    def _build_dynamic_pipeline(self, df: pd.DataFrame, target: str):
        """Pipeline dinámico con manejo de tipos de datos"""
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.impute import SimpleImputer

        # Identificar tipos de columnas (excluyendo el target)
        numeric_features = df.drop(columns=[target]).select_dtypes(include=np.number).columns
        categorical_features = df.drop(columns=[target]).select_dtypes(exclude=np.number).columns

        # Construir transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Se fuerza a OneHotEncoder a devolver un array denso (sparse=False)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        return preprocessor

    def _execute_pipeline(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Ejecuta el preprocesamiento y devuelve datos transformados"""
        X = df.drop(columns=[target])
        y = df[target].copy()
        
        if self.pipeline:
            X_transformed = self.pipeline.fit_transform(X)
            # Si el resultado es disperso, lo convertimos a denso.
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            # Intentamos obtener nombres de características
            if hasattr(self.pipeline, 'get_feature_names_out'):
                features = self.pipeline.get_feature_names_out()
                # Solo asignamos los nombres si la cantidad coincide
                if X_transformed.ndim == 2 and X_transformed.shape[1] == len(features):
                    X_df = pd.DataFrame(X_transformed, columns=features)
                else:
                    X_df = pd.DataFrame(X_transformed)
            else:
                X_df = pd.DataFrame(X_transformed)
            return X_df, y
        
        return X, y

class ModelShamanAgent(BaseAgent):
    """
    Agente encargado de:
    - Generar y compilar modelos automáticamente
    - Validar el modelo generado (entrenamiento rápido + métricas)
    - Proveer fallback en caso de errores
    """
    def __init__(self, ctx: OperationalContext):
        super().__init__("ModelShaman", ctx)
        self.current_model = None
        # Inicializamos el generador de código
        self.genai = genai.GenerativeModel('gemini-pro')
    
    @resilient_agent(max_retries=5)
    def conjure_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Generación de modelos con auto-corrección y validación integrada."""
        print("Iniciando conjuración del modelo...")
        problem_type = self._detect_problem_type(y)
        
        for attempt in range(3):
            try:
                model_code = self._generate_model_code(X, y, problem_type)
                # Ahora pasamos problem_type a _compile_model para seleccionar el modelo adecuado
                model = self._compile_model(model_code, problem_type)
                self._validate_model(model, X, y, problem_type)
                self.current_model = model
                print("Modelo generado y validado con éxito.")
                return model
            except Exception as e:
                self.log(f"Intento {attempt+1} fallido al generar modelo: {str(e)}", ErrorSeverity.WARNING)
                self.ctx.log_error({
                    'phase': 'model_generation',
                    'error': str(e),
                    'attempt': attempt+1
                })
        
        self.log("Fallback a modelo baseline", ErrorSeverity.CRITICAL)
        fallback_model = self._create_fallback_model(problem_type)
        self.current_model = fallback_model
        print("Utilizando modelo baseline como fallback.")
        return fallback_model

    def _detect_problem_type(self, y: pd.Series) -> str:
        """Diferencia automáticamente entre problema de clasificación o regresión."""
        if pd.api.types.is_numeric_dtype(y):
            return 'regression'
        else:
            return 'classification'

    def _generate_model_code(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> str:
        """Generación de código con validación semántica"""
        context_info = f"""
        Dataset shape: {X.shape}
        Tipos de datos:
        - Numéricas: {X.select_dtypes(include=np.number).columns.tolist()}
        - Categóricas: {X.select_dtypes(exclude=np.number).columns.tolist()}
        Target type: {y.dtype}
        Problem type: {problem_type}
        """
        
        prompt = f"""
        Genera código Python robusto para un problema de {problem_type} considerando:
        {context_info}
        
        Requisitos obligatorios:
        1. Manejo automático de características categóricas
        2. Validación de tipos de datos
        3. Transformación adecuada del target si es necesario
        4. Pipeline completo con preprocesamiento integrado
        
        Ejemplo para clasificación categórica:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
        
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(), categorical_features)]
        )
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ])
        """
        response = self.genai.generate_content(prompt)
        return self._extract_validated_code(response.text)

    def _create_intelligent_prompt(self, context: str, problem_type: str) -> str:
        """Generación de prompts adaptativos basados en el contexto."""
        return f"""
        Genera código Python robusto para un problema de {problem_type}.
        Considera: {context}
        Requisitos:
        - Validación cruzada estratificada (si es clasificación)
        - Manejo de clases desbalanceadas
        - Early stopping automático
        - Registro de métricas detalladas
        - Guardado seguro del modelo
        """

    def _extract_validated_code(self, text: str) -> str:
        """
        Se podría parsear y validar sintácticamente el código generado.
        Para simplificar, lo devolvemos directamente.
        """
        return text

    def _compile_model(self, model_code: str, problem_type: str):
        """
        Simula la compilación de un modelo a partir de 'model_code'.
        En un escenario real, se podría usar 'exec' para obtener un objeto Python,
        o retornar directamente un modelo predefinido de sklearn.
        Aquí, seleccionamos un modelo base según el problema.
        """
        if problem_type == 'classification':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression()
        else:
            from sklearn.linear_model import LinearRegression
            return LinearRegression()

    def _validate_model(self, model, X: pd.DataFrame, y: pd.Series, problem_type: str):
        """
        Validación mejorada del modelo:
        - Verificación de tipos de datos en X
        - Transformación de y si es de clasificación y tiene tipo 'object'
        - Split train/test
        - Entrena el modelo
        - Calcula métricas
        - Imprime y registra métricas en el contexto
        """
        if len(X) == 0:
            raise ValueError("El dataset está vacío. No se puede entrenar el modelo.")

        # Verificar que todas las características sean numéricas
        if X.select_dtypes(exclude=np.number).shape[1] > 0:
            raise ValueError("Existen características no numéricas sin preprocesar.")

        # Transformar target si es de clasificación y es categórico
        if problem_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)

        print("Validando modelo con train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluación del modelo
        if problem_type == 'classification':
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            print(f"  - Accuracy: {acc:.3f}")
            print(f"  - F1 Score: {f1:.3f}")
            self.ctx._state['performance_metrics']['accuracy'] = acc
            self.ctx._state['performance_metrics']['f1_score'] = f1
        else:
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            print(f"  - R2 Score: {r2:.3f}")
            print(f"  - MSE: {mse:.3f}")
            self.ctx._state['performance_metrics']['r2'] = r2
            self.ctx._state['performance_metrics']['mse'] = mse

    def _create_fallback_model(self, problem_type: str) -> Any:
        """
        Retorna un modelo baseline en caso de fallos repetidos.
        """
        from sklearn.dummy import DummyClassifier, DummyRegressor
        if problem_type == 'classification':
            return DummyClassifier(strategy='most_frequent')
        else:
            return DummyRegressor(strategy='mean')

class OracleAgent(BaseAgent):
    """
    Agente de validación y toma de decisiones finales.
    """
    def __init__(self, ctx: OperationalContext):
        super().__init__("Oracle", ctx)
    
    def validate_workflow(self) -> bool:
        """Validación integral de todo el workflow."""
        print("Validando el flujo completo en OracleAgent...")
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
        
        print("Validación del flujo completada correctamente.")
        return True
    
    def _validate_data_integrity(self) -> bool:
        """Validación de la trazabilidad de datos (placeholder)."""
        return True
    
    def _validate_model_performance(self) -> bool:
        """Chequear si el modelo cumple métricas mínimas (placeholder)."""
        return True
    
    def _validate_deployment_readiness(self) -> bool:
        """Confirma que los pasos del pipeline estén OK para deploy (placeholder)."""
        return True

class PhoenixAgent(BaseAgent):
    """
    Agente de recuperación de fallos catastróficos.
    """
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
        """
        Estrategias avanzadas de recuperación (placeholder):
        1. Rollback a versión estable
        2. Regeneración de datos intermedios
        3. Reentrenamiento con parámetros conservadores
        4. Recolección automática de datos adicionales
        """
        return True

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
            print("Cargando datos...")
            self.ctx.update_stage(WorkflowStage.DATA_LOADING)
            raw_data = self._load_data()
            print("Datos cargados correctamente.")

            # Fase 1.1: Análisis de intención
            intent = self.agents['intent'].analyze_intent(user_input)

            # Fase 1.2: Resolución de la columna objetivo
            target_col = self.agents['data_guardian'].resolve_target(raw_data, intent)
            self.ctx._state['target_column'] = target_col
            self.ctx.update_stage(WorkflowStage.DATA_VALIDATION)
            
            # Fase 2: Preprocesamiento adaptativo
            clean_data, target = self.agents['data_alchemist'].auto_preprocess(raw_data)
            self.ctx.update_stage(WorkflowStage.FEATURE_ENGINEERING)
            
            # Fase 3: Modelado inteligente
            model = self.agents['model_shaman'].conjure_model(clean_data, target)
            self.ctx.update_stage(WorkflowStage.MODEL_VALIDATION)
            
            # Fase 4: Validación final y deployment
            if self.agents['oracle'].validate_workflow():
                self._deploy_model(model)
                return True
            
            return False
        
        except Exception as e:
            # Actualizamos la etapa a ERROR_HANDLING para que Phoenix actúe
            self.ctx.update_stage(WorkflowStage.ERROR_HANDLING)
            self.ctx.log_error({'phase': 'global', 'error': str(e)})
            # Intento de recuperación
            if self.agents['phoenix'].resurrect_workflow():
                return self.execute_workflow(user_input)
            return False
    
    def _load_data(self) -> pd.DataFrame:
        """
        Carga de datos con múltiples posibles fuentes/formatos (placeholder).
        En un caso real, se usaría un DataLoader robusto.
        """
        if not os.path.exists('dataset.csv'):
            raise FileNotFoundError("dataset.csv no existe en el directorio actual.")
        return pd.read_csv('dataset.csv')
    
    def _deploy_model(self, model):
        """Implementar lógica de deployment (placeholder)."""
        print("Desplegando el modelo a producción...")
        joblib.dump(model, 'model.joblib')
        self.ctx.update_stage(WorkflowStage.DEPLOYMENT)
        print("¡Modelo desplegado con éxito!")

# ---------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    cortex = AICortex()
    user_input = input("Describe tu tarea de ML (ej: 'predecir riesgo de diabetes'): ")
    
    if cortex.execute_workflow(user_input):
        print("\n🔥 Workflow completado con éxito!")
        print("Modelo deployado y listo para producción")
    else:
        print("\n🛑 Workflow fallido después de múltiples intentos")
        print("Reporte de errores:")
        for error in cortex.ctx.get_context()['error_log']:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(error['timestamp']))
            print(f"[{ts}] {error.get('agent', 'N/A')}: {error.get('message')}")
