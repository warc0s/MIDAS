from autogen import ConversableAgent
import joblib
import os
from sklearn.pipeline import Pipeline

# Configuración del modelo LLM
llm_config = { 
    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "base_url": "https://api.deepinfra.com/v1/openai",
    "api_key": os.getenv("DEEPINFRA_KEY"),
    "temperature": 0.7,
    "seed": 42
}

# Definir agentes en secuencia
model_analyzer = ConversableAgent(
    name="Model_Analyzer",
    llm_config=llm_config,
    system_message="You analyze machine learning models stored in joblib files and provide a summary.",
    description="Analyzes ML models and provides a report.",
)

ui_designer = ConversableAgent(
    name="UI_Designer",
    llm_config=llm_config,
    system_message="You design a Streamlit UI based on the model analysis.",
    description="Creates a UI design based on model insights.",
)

code_generator = ConversableAgent(
    name="Code_Generator",
    llm_config=llm_config,
    system_message="You implement the UI design into a working Streamlit application. When you end you have to say 'CODE GENERATION COMPLETE'.",
    description="Generates the Streamlit code for the app.",
)

# Usuario que orquesta la secuencia
user_proxy = ConversableAgent(
    name="User_Proxy",
    description="Monitors the conversation and ensures sequential execution.",
    llm_config=llm_config
)

# Procesar archivo joblib
def process_joblib(file_path):
    try:
        model = joblib.load(file_path)
        if isinstance(model, Pipeline):
            steps = [step[0] for step in model.steps]
            last_step = model.steps[-1][1]
            features = getattr(last_step, "feature_names_in_", ["feature1", "feature2"])
        else:
            steps = ["Not a pipeline"]
            features = getattr(model, "feature_names_in_", ["feature1", "feature2"])

        return {
            "type": type(model).__name__,
            "pipeline_steps": steps,
            "features": features
        }
    except Exception as e:
        return {"error": f"Error loading model: {str(e)}"}

# Flujo de conversación secuencial
def start_conversation(model_info):
    chat_results = user_proxy.initiate_chats(
        [
            {
                "recipient": model_analyzer,
                "message": f"Analyze this machine learning model: {model_info}",
                "max_turns": 1,
                "summary_method": "last_msg",
            },
            {
                "recipient": ui_designer,
                "message": "Based on the analysis, design a Streamlit UI.",
                "max_turns": 1,
                "summary_method": "last_msg",
            },
            {
                "recipient": code_generator,
                "message": "Generate the final Streamlit application.",
                "max_turns": 1,
                "summary_method": "last_msg",
            },
        ]
    )

    # Imprimir resúmenes de cada etapa
    print("\n\nModel Analysis Summary:\n", chat_results[0].summary)
    print("\n\nUI Design Summary:\n", chat_results[1].summary)
    print("\n\nCode Generation Summary:\n", chat_results[2].summary)

# Main
def main():
    file_path = input("Ingrese la ruta del archivo joblib: ").strip()

    if not os.path.exists(file_path):
        print("Error: El archivo no existe. Verifique la ruta e intente nuevamente.")
        return

    model_info = process_joblib(file_path)
    if "error" in model_info:
        print(model_info["error"])
        return

    start_conversation(model_info)

if __name__ == "__main__":
    main()
