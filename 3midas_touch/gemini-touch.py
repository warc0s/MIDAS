import os
import sys
import logging
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
import joblib
import re
import ast
import time

# ---------------------------------------------------------------------
# Configuración inicial
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(agent)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()
genai.configure(api_key=os.getenv("google_key"))

# ---------------------------------------------------------------------
# Agente: Logger
# ---------------------------------------------------------------------
class LoggerAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.logger = logging.getLogger(__name__)

    def info(self, message):
        self.logger.info(message, extra={'agent': self.agent_name})

    def warning(self, message):
        self.logger.warning(message, extra={'agent': self.agent_name})

    def error(self, message):
        self.logger.error(message, extra={'agent': self.agent_name})
        
    def debug(self, message):
        self.logger.debug(message, extra={'agent': self.agent_name})

# ---------------------------------------------------------------------
# Agente 1: Data Loader
# ---------------------------------------------------------------------
class DataLoaderAgent:
    def __init__(self):
        self.logger = LoggerAgent("DataLoaderAgent")

    def load_and_validate(self, file_path='dataset.csv', min_rows=10, min_cols=2):
        """Carga y valida la estructura básica del dataset"""
        try:
            start_time = time.time()
            self.logger.info(f"Iniciando carga de datos desde '{file_path}'")
            df = pd.read_csv(file_path)
            elapsed_time = time.time() - start_time
            self.logger.info(f"Datos cargados en {elapsed_time:.2f} segundos. Dimensiones: {df.shape}")

            # Validación básica
            if len(df) < min_rows:
                self.logger.warning(f"Dataset con muy pocas filas ({len(df)} < {min_rows})")

            if len(df.columns) < min_cols:
                self.logger.error(f"Dataset con muy pocas columnas ({len(df.columns)} < {min_cols})")
                raise ValueError("Dataset insuficiente para modelado")

            self.logger.info("Validación de dataset exitosa")
            return df
        except FileNotFoundError:
            self.logger.error(f"Archivo no encontrado: '{file_path}'")
            raise
        except pd.errors.ParserError:
            self.logger.error(f"Error al parsear el archivo CSV: '{file_path}'")
            raise
        except Exception as e:
            self.logger.error(f"Error inesperado al cargar datos: {e}")
            raise

# ---------------------------------------------------------------------
# Agente 2: Data Inspector
# ---------------------------------------------------------------------
class DataInspectorAgent:
    def __init__(self):
        self.logger = LoggerAgent("DataInspectorAgent")

    def analyze_dataset(self, df, target_column):
        """Analiza el dataset, detecta problemas y sugiere el tipo de problema"""
        self.logger.info("Iniciando análisis del dataset")
        report = {
            'missing_values': {},
            'data_types': {},
            'outliers': {},
            'target_distribution': None,
            'problem_type': None
        }

        # Verificar existencia de columna target
        if target_column not in df.columns:
            self.logger.error(f"Columna target '{target_column}' no encontrada")
            raise ValueError(f"Columna target '{target_column}' no encontrada")

        # Análisis de valores faltantes
        missing_values = df.isnull().sum()
        report['missing_values'] = {col: val for col, val in missing_values.items() if val > 0}
        self.logger.info(f"Valores faltantes detectados: {report['missing_values']}")

        # Análisis de tipos de datos
        report['data_types'] = df.dtypes.apply(lambda x: x.name).to_dict()
        self.logger.info(f"Tipos de datos: {report['data_types']}")

        # Detección de outliers (solo numéricas)
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col != target_column:
                q1, q3 = np.percentile(df[col], [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if not outliers.empty:
                    report['outliers'][col] = outliers.tolist()
        self.logger.info(f"Outliers detectados en: {list(report['outliers'].keys())}")

        # Análisis de la distribución del target y definición del tipo de problema
        if np.issubdtype(df[target_column].dtype, np.number):
            report['target_distribution'] = df[target_column].describe().to_dict()
            report['problem_type'] = 'regression'
            self.logger.info("Problema de REGRESIÓN detectado")
        else:
            report['target_distribution'] = df[target_column].value_counts(normalize=True).to_dict()
            report['problem_type'] = 'classification'
            self.logger.info("Problema de CLASIFICACIÓN detectado")

        self.logger.info("Análisis de dataset completado")
        return report

# ---------------------------------------------------------------------
# Agente 3: Preprocessing Architect
# ---------------------------------------------------------------------
class PreprocessingArchitectAgent:
    def __init__(self):
        self.logger = LoggerAgent("PreprocessingArchitectAgent")

    def design_pipeline(self, df, report, target_column):
        """Diseña un pipeline de preprocesamiento basado en el análisis"""
        self.logger.info("Diseñando pipeline de preprocesamiento")
        numeric_features = []
        categorical_features = []

        for col, dtype in report['data_types'].items():
            if col != target_column:
                if 'int' in dtype or 'float' in dtype:
                    numeric_features.append(col)
                else:
                    categorical_features.append(col)

        # Transformaciones para variables numéricas
        numeric_transformer = []
        if any(col in report['missing_values'] for col in numeric_features):
            self.logger.info("Imputación de valores faltantes para variables numéricas")
            numeric_transformer.append(('imputer', SimpleImputer(strategy='median')))
        if any(col in report['outliers'] for col in numeric_features):
            self.logger.info("Escalado robusto para variables numéricas")
            numeric_transformer.append(('scaler', StandardScaler()))

        # Transformaciones para variables categóricas
        categorical_transformer = []
        if any(col in report['missing_values'] for col in categorical_features):
            self.logger.info("Imputación de valores faltantes para variables categóricas")
            categorical_transformer.append(('imputer', SimpleImputer(strategy='most_frequent')))
        self.logger.info("Codificación One-Hot para variables categóricas")
        categorical_transformer.append(('onehot', OneHotEncoder(handle_unknown='ignore')))

        # Combinar transformaciones en un ColumnTransformer
        transformers = []
        if numeric_transformer:
            transformers.append(('num', Pipeline(steps=numeric_transformer), numeric_features))
        if categorical_transformer:
            transformers.append(('cat', Pipeline(steps=categorical_transformer), categorical_features))

        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        self.logger.info("Pipeline de preprocesamiento diseñado")
        return preprocessor

# ---------------------------------------------------------------------
# Agente 4: Preprocessing Executor
# ---------------------------------------------------------------------
class PreprocessingExecutorAgent:
    def __init__(self):
        self.logger = LoggerAgent("PreprocessingExecutorAgent")

    def apply_pipeline(self, df, pipeline, target_column):
        """Aplica el pipeline de preprocesamiento al dataset"""
        self.logger.info("Aplicando pipeline de preprocesamiento")
        try:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            X_transformed = pipeline.fit_transform(X)

            # Convertir sparse matrix a dense array si es necesario
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            
            # Obtener nombres de características
            feature_names = self._get_feature_names(pipeline)

            # Validar dimensiones
            if X_transformed.ndim == 1:
                X_transformed = X_transformed.reshape(-1, 1)
                
            # Asegurar consistencia en el número de características
            if X_transformed.shape[1] != len(feature_names):
                self.logger.error(
                    f"Discrepancia en dimensiones: "
                    f"Datos transformados {X_transformed.shape} vs "
                    f"Nombres de características ({len(feature_names)})"
                )
                raise ValueError("Inconsistencia en features transformados")

            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
            self.logger.info("Pipeline aplicado exitosamente")
            return X_transformed, y
        except Exception as e:
            self.logger.error(f"Error al aplicar el pipeline: {e}")
            raise

    def _get_feature_names(self, pipeline):
        """Obtiene nombres de características usando el método nativo de scikit-learn"""
        try:
            return pipeline.get_feature_names_out()
        except AttributeError:
            return self._legacy_get_feature_names(pipeline)
    
    def _legacy_get_feature_names(self, pipeline):
        """Método alternativo para versiones antiguas de scikit-learn"""
        feature_names = []
        for name, trans, columns in pipeline.transformers_:
            if trans == 'drop':
                continue
            if hasattr(trans, 'get_feature_names_out'):
                names = trans.get_feature_names_out(columns)
            else:
                names = columns
            feature_names.extend(names)
        return feature_names

# ---------------------------------------------------------------------
# Agente 5: Model Architect (Gemini-powered)
# ---------------------------------------------------------------------
class ModelArchitectAgent:
    def __init__(self):
        self.logger = LoggerAgent("ModelArchitectAgent")
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def generate_training_code(self, context, problem_type, target_column, max_retries=3, retry_delay=10):
        """Genera código de entrenamiento usando Gemini, con reintentos"""
        context += f"\nIMPORTANTE: Usar y_train/y_test directamente, NO hacer referencia a '{target_column}' en el código."
        prompt = self._create_prompt(context, problem_type, target_column)
        self.logger.info("Generando código de entrenamiento con Gemini")

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.model.generate_content(prompt)
                elapsed_time = time.time() - start_time
                code = self._extract_code(response.text)
                self.logger.info(f"Código generado en {elapsed_time:.2f} segundos. Intento {attempt + 1}/{max_retries}")

                if self._validate_code(code, problem_type, target_column):
                    self.logger.info("Código de entrenamiento generado y validado exitosamente")
                    return code
                else:
                    self.logger.warning("Validación de código fallida. Reintentando...")

            except Exception as e:
                self.logger.error(f"Error en generación de código (intento {attempt + 1}): {e}")

            time.sleep(retry_delay)

        self.logger.error("Número máximo de reintentos alcanzado. Fallo en generación de código.")
        raise RuntimeError("Fallo en generación de código después de múltiples reintentos")

    def _create_prompt(self, context, problem_type, target_column):
        return f"""
        Eres un experto en Machine Learning. Genera código Python para resolver un problema de {problem_type}.
        
        El dataset ya está preprocesado y dividido en:
        - X_train.csv: Features de entrenamiento
        - X_test.csv: Features de test
        - y_train.csv: Target de entrenamiento
        - y_test.csv: Target de test

        Requerimientos:
        1. Cargar los datos usando:
           X_train = pd.read_csv('X_train.csv')
           y_train = pd.read_csv('y_train.csv').squeeze()  # Convertir a Series
        2. Entrenar modelo con .fit(X_train, y_train)
        3. Evaluar con y_test (usar y_test = pd.read_csv('y_test.csv').squeeze())
        4. No referenciar '{target_column}' directamente (ya está separado en y_train/y_test)
        5. Guarda el modelo entrenado como 'model.joblib'.
        6. Genera un reporte de clasificación (si es clasificación) o métricas de regresión (si es regresión).
        7. El código debe ser auto-contenido, eficiente y robusto.
        8. Utiliza bibliotecas estándar de Python para ML (e.g., scikit-learn).
        9. Incluye validación cruzada si es apropiado.
        10. Asegúrate de que el código maneje correctamente diferentes tipos de datos y missing values.
        11. No incluyas visualizaciones.
        12. Comenta el código para explicar cada paso.

        Escribe solamente el código Python, sin texto adicional.
        """

    def _extract_code(self, text):
        """Extrae el código Python de la respuesta de Gemini"""
        code = text.split("```python")[1].split("```")[0].strip()
        return code

    def _validate_code(self, code, problem_type, target_column):
        """Valida el código generado de forma más precisa"""
        # Validación de importaciones
        if not re.search(r"from\s+sklearn", code):
            self.logger.warning("No se importan módulos de scikit-learn")
            return False

        # Validación de uso del target (nueva lógica)
        y_usage = (
            re.search(r"y_train\s*=\s*", code) and 
            re.search(r"y_test\s*=\s*", code) and
            re.search(r"\.fit\(.*y_train", code)
        )
        
        if not y_usage:
            self.logger.warning("No se detectó uso correcto de la variable target")
            return False

        # Validación de tipo de problema
        if problem_type == 'classification' and "classification_report" not in code:
            self.logger.warning("Falta reporte de clasificación")
            return False
            
        if problem_type == 'regression' and not any(metric in code for metric in ["mean_squared_error", "r2_score"]):
            self.logger.warning("Faltan métricas de regresión")
            return False

        # Validación de guardado del modelo
        if "joblib.dump" not in code:
            self.logger.warning("No se guarda el modelo")
            return False

        return True

# ---------------------------------------------------------------------
# Agente 6: Model Trainer
# ---------------------------------------------------------------------
class ModelTrainerAgent:
    def __init__(self):
        self.logger = LoggerAgent("ModelTrainerAgent")

    def execute_training(self, code, X_train, X_test, y_train, y_test):
        """Ejecuta el código de entrenamiento"""
        self.logger.info("Ejecutando código de entrenamiento")
        
        # Guardar los dataframes como csv
        X_train.to_csv('X_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)
        
        # Espacio de nombres para capturar resultados
        local_vars = {
            'report': None,
            'model': None
        }

        try:
            start_time = time.time()
            exec(code, {'__builtins__': __builtins__}, local_vars)
            elapsed_time = time.time() - start_time
            self.logger.info(f"Entrenamiento completado en {elapsed_time:.2f} segundos")

            report = local_vars['report']
            model = local_vars['model']

            if not model:
                self.logger.error("No se ha generado un modelo entrenado")
                raise ValueError("No se generó un modelo durante el entrenamiento")

            if report:
                self.logger.info("Reporte de evaluación generado")
                print("\n📊 Reporte de evaluación:")
                print(report)
            else:
                self.logger.warning("No se generó reporte de evaluación")

            return model, report

        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento: {e}")
            raise

# ---------------------------------------------------------------------
# Agente 7: Model Evaluator
# ---------------------------------------------------------------------
class ModelEvaluatorAgent:
    def __init__(self):
        self.logger = LoggerAgent("ModelEvaluatorAgent")

    def evaluate_model(self, model, X_test, y_test, problem_type):
        """Evalúa el modelo y genera métricas adicionales"""
        self.logger.info("Evaluando modelo")
        try:
            y_pred = model.predict(X_test)

            if problem_type == 'classification':
                report = classification_report(y_test, y_pred)
                self.logger.info("Reporte de clasificación generado")
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                report = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR2 Score: {r2:.4f}"
                self.logger.info("Métricas de regresión calculadas")

            return report
        except Exception as e:
            self.logger.error(f"Error durante la evaluación del modelo: {e}")
            raise

# ---------------------------------------------------------------------
# Workflow Principal
# ---------------------------------------------------------------------
def ml_workflow():
    try:
        # Paso 1: Obtener input del usuario
        user_input = input("Ingrese su tarea de ML (ej: 'predecir columna calories'): ")
        if not user_input:
            raise ValueError("La entrada del usuario no puede estar vacía")
        target_column = user_input.split()[-1]

        # Paso 2: Cargar datos
        data_loader = DataLoaderAgent()
        df = data_loader.load_and_validate()

        # Paso 3: Análisis inicial
        inspector = DataInspectorAgent()
        report = inspector.analyze_dataset(df, target_column)
        problem_type = report['problem_type']

        # Paso 4: Diseño del pipeline de preprocesamiento
        preprocessor_architect = PreprocessingArchitectAgent()
        pipeline = preprocessor_architect.design_pipeline(df, report, target_column)

        # Paso 5: Aplicar pipeline de preprocesamiento
        preprocessor_executor = PreprocessingExecutorAgent()
        X_transformed, y = preprocessor_executor.apply_pipeline(df, pipeline, target_column)

        # Paso 6: Dividir datos en train/test
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

        # Paso 7: Generar código de entrenamiento
        model_architect = ModelArchitectAgent()
        context = f"""
        Dataset con {X_train.shape[0]} filas y {X_train.shape[1]} columnas.
        Columnas: {', '.join(X_train.columns)}
        Tipos de datos: {report['data_types']}
        Valores faltantes: {report['missing_values']}
        Outliers detectados en: {list(report['outliers'].keys()) if report['outliers'] else 'Ninguna'}
        """
        training_code = model_architect.generate_training_code(context, problem_type, target_column)

        # Paso 8: Ejecutar entrenamiento
        trainer = ModelTrainerAgent()
        model, training_report = trainer.execute_training(training_code, X_train, X_test, y_train, y_test)

        # Paso 9: Evaluar modelo y generar reporte adicional
        evaluator = ModelEvaluatorAgent()
        evaluation_report = evaluator.evaluate_model(model, X_test, y_test, problem_type)

        # Paso 10: Resultados
        print("\n📊 Reporte de evaluación adicional:")
        print(evaluation_report)

    except Exception as e:
        logging.error(f"❌ Error en el workflow: {e}")

if __name__ == "__main__":
    ml_workflow()