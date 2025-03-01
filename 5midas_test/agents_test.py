from autogen import ConversableAgent, GroupChat, GroupChatManager
import joblib
import os
import time
import numpy as np
import psutil
import datetime
from deep_translator import GoogleTranslator
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# Configuración del modelo LLM
llm_config = { 
    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "base_url": "https://api.deepinfra.com/v1/openai",
    "api_key": os.getenv("DEEPINFRA_KEY"),
    "temperature": 0.7,
    "seed": 42
}

# Definición de agentes con ConversableAgent
model_analyzer = ConversableAgent(
    name="Model Analyzer",
    llm_config=llm_config,
    system_message="You analyze machine learning models stored in joblib files and provide a summary. Don't write code",
    description="Analyzes ML models and provides a report.",
    is_termination_msg= lambda msg: "See you soon!" in (msg.get("content") or "")
)

performance_tester = ConversableAgent(
    name="Performance Tester",
    llm_config=llm_config,
    system_message="You test the performance of ML models, including latency, memory usage, and CPU usage. Don't write code.",
    description="Measures latency, memory, and CPU usage."
)

robustness_checker = ConversableAgent(
    name="Robustness Checker",
    llm_config=llm_config,
    system_message="You evaluate how robust an ML model is against null values, incorrect types, and extreme values. Don't write code.",
    description="Checks model robustness against various inputs."
)

output_validator = ConversableAgent(
    name="Output Validator",
    llm_config=llm_config,
    system_message="You validate the correctness of model predictions, ensuring they are in the expected format and range. End your responses with 'See you soon!'",
    description="Validates output format and correctness."
)

# Crear GroupChat para la comunicación entre agentes
groupchat = GroupChat(
    agents=[model_analyzer, performance_tester, robustness_checker, output_validator],
    speaker_selection_method="round_robin",
    messages=[]
)

# Crear GroupChatManager para coordinar la conversación
group_manager = GroupChatManager(
    name="group_manager",
    groupchat=groupchat,
    llm_config=llm_config,
)

def load_model(file_path):
    """Carga el modelo joblib y mide el tiempo de carga."""
    if not file_path.endswith(".joblib") or not os.path.isfile(file_path):
        return None, None, None
    start_time = time.time()
    try:
        model = joblib.load(file_path)
        load_time = time.time() - start_time
        size_on_disk = os.path.getsize(file_path) / (1024 * 1024)
        return model, load_time, size_on_disk
    except Exception:
        return None, None, None
    
def check_model_validity(model):
    """Verifica si el modelo es un estimador de Scikit-Learn."""
    return isinstance(model, (Pipeline, BaseEstimator))


def measure_latency(model, X_sample, batch_sizes=[1, 100, 1000, 10000]):
    """Mide la latencia en diferentes tamaños de batch."""
    latencies = {}
    for batch in batch_sizes:
        X_batch = np.repeat(X_sample, batch, axis=0) if X_sample.ndim == 2 else np.tile(X_sample, (batch, 1))
        start_time = time.time()
        model.predict(X_batch)
        latencies[batch] = (time.time() - start_time) * 1000 / batch
    return latencies

def measure_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def measure_memory_and_cpu_during_prediction(model, X_sample):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    cpu_before = psutil.cpu_percent(interval=None)

    start_time = time.time()
    model.predict(X_sample)
    end_time = time.time()

    mem_after = process.memory_info().rss / (1024 * 1024)
    cpu_after = psutil.cpu_percent(interval=None)

    return {
        "memory_peak": mem_after - mem_before,
        "cpu_usage": cpu_after - cpu_before,
        "prediction_time": (end_time - start_time) * 1000  # en ms
    }


def validate_predictions(model, X_sample):
    predictions = model.predict(X_sample)
    return {
        "valid_format": isinstance(predictions, np.ndarray),
        "valid_range": (np.min(predictions), np.max(predictions)),
        "probabilities_sum_1": np.allclose(np.sum(predictions, axis=1), 1) if len(predictions.shape) > 1 else None
    }

def check_robustness(model, X_sample):
    robustness_tests = {}
    try:
        robustness_tests["null_values"] = model.predict(np.full_like(X_sample, np.nan))
    except:
        robustness_tests["null_values"] = "Failed"
    try:
        robustness_tests["out_of_range"] = model.predict(X_sample * 1000)
    except:
        robustness_tests["out_of_range"] = "Failed"
    try:
        robustness_tests["wrong_data_type"] = model.predict(X_sample.astype(str))
    except:
        robustness_tests["wrong_data_type"] = "Failed"

    # Consistencia de predicciones
    pred1 = model.predict(X_sample)
    pred2 = model.predict(X_sample)
    robustness_tests["consistent_predictions"] = np.allclose(pred1, pred2)

    return robustness_tests

def process_joblib(file_path):
    model, load_time, size_on_disk = load_model(file_path)
    if model is None:
        return {"error": "Error loading model."}
    
    validity = check_model_validity(model)
    num_features = getattr(model, "n_features_in_", 2)
    X_sample = np.random.rand(1, num_features)
    latencies = measure_latency(model, X_sample)
    performance_metrics = measure_memory_and_cpu_during_prediction(model, X_sample)
    memory_usage = measure_memory_usage()
    predictions_validity = validate_predictions(model, X_sample)
    robustness = check_robustness(model, X_sample)
    
    throughput = 1000 / latencies[1000]  # Predicciones por segundo en batch de 1000
   
    return {
        "load_time": load_time,
        "size_on_disk": size_on_disk,
        "valid_model": validity,
        "latencies": latencies,
        "memory_usage": memory_usage,
        "performance_metrics": performance_metrics,
        "predictions_validity": predictions_validity,
        "robustness_tests": robustness,
        "throughput": throughput,
        "final_recommendation": "APTO" if validity and robustness["consistent_predictions"] else "NO APTO"
    }



def translate_to_spanish(text):
    return GoogleTranslator(source="en", target="es").translate(text)

def generate_markdown_report(messages):
    """Extrae la información de los mensajes, los clasifica antes de traducir y genera un informe en español."""
    analysis_results = {
        "overview": "",
        "performance": "",
        "latency": "",
        "validity": "",
        "robustness": "",
        "recommendation": "",
        "improvements": "",
    }

    # Primero clasificamos los mensajes en inglés
    for message in messages:
        content = message["content"]
        if "Model Overview" in content:
            analysis_results["overview"] = content
        elif "Performance Metrics" in content:
            analysis_results["performance"] = content
        elif "Latency Analysis" in content:
            analysis_results["latency"] = content
        elif "Predictions Validity" in content:
            analysis_results["validity"] = content
        elif "Robustness Tests" in content:
            analysis_results["robustness"] = content
        elif "Final Recommendation" in content:
            analysis_results["recommendation"] = content
        elif "Recommendations for Improvement" in content:
            analysis_results["improvements"] = content

    # Ahora traducimos cada sección
    for key in analysis_results:
        if analysis_results[key]:
            analysis_results[key] = translate_to_spanish(analysis_results[key])

    # Generar el informe en español
    sections = [
        ("## 🔍 Resumen del Modelo", analysis_results["overview"]),
        ("## ⚙️ Métricas de Rendimiento", analysis_results["performance"]),
        ("## ⏳ Análisis de Latencia", analysis_results["latency"]),
        ("## ✅ Validez de Predicciones", analysis_results["validity"]),
        ("## 🛡️ Pruebas de Robustez", analysis_results["robustness"]),
        ("## 📌 Recomendación Final", f"**{analysis_results['recommendation']}**" if analysis_results["recommendation"] else ""),
        ("## 🔧 Sugerencias de Mejora", analysis_results["improvements"])
    ]

    markdown_content = f"# 📊 Informe de Análisis del Modelo\n"
    markdown_content += f"**Generado el:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown_content += "---\n\n"

    for title, content in sections:
        if content.strip():
            markdown_content += f"{title}\n{content}\n\n"

    file_name = "informe_analisis_modelo.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(markdown_content.strip())

    print(f"📄 Informe guardado como: {file_name}")
    return "¡Informe guardado!"


def main():
    file_path = input("Ingrese la ruta del archivo joblib: ").strip()
    if not os.path.exists(file_path):
        print("Error: El archivo no existe. Verifique la ruta e intente nuevamente.")
        return
    
    model_info = process_joblib(file_path)
    if "error" in model_info:
        print(model_info["error"])
        return

    print("⏳ Ejecutando análisis del modelo...")

    # Iniciar conversación con los agentes (máximo una ronda por agente)
    group_manager.initiate_chat(
        recipient=group_manager,
        max_turns=len(groupchat.agents),  # Se asegura de que cada agente hable solo una vez
        message=f"Analyze the following ML model: {model_info} and generate a Markdown report.",
        )

    print("✅ Todos los agentes han respondido. Finalizando ejecución...")

    # Generar el reporte si hay mensajes
    if groupchat.messages:
        generate_markdown_report(groupchat.messages)
        print("📄 Reporte generado exitosamente.")
    else:
        print("⚠️ No hay mensajes en el chat de grupo. No se generó reporte.")

if __name__ == "__main__":
    main()
