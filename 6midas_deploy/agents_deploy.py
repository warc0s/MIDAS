from autogen import ConversableAgent
import joblib
import os
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

def get_model_info(model):
    """Extrae información relevante de un modelo de machine learning cargado."""
    try:
        model_type = type(model).__name__
        model_info = {"type": model_type, "details": {}}

        if isinstance(model, Pipeline):
            model_info["pipeline_steps"] = [step[0] for step in model.steps]
            last_step = model.steps[-1][1]
        else:
            last_step = model

        if hasattr(last_step, "feature_names_in_"):
            model_info["features"] = list(last_step.feature_names_in_)
            model_info["num_features"] = len(last_step.feature_names_in_)
        elif hasattr(last_step, "n_features_in_"):
            model_info["num_features"] = last_step.n_features_in_
            model_info["features"] = "Unknown (feature names not available)"
        else:
            model_info["features"] = "Unknown"
            model_info["num_features"] = "Unknown"

        if isinstance(last_step, BaseEstimator):
            model_info["details"]["params"] = last_step.get_params()

        return model_info
    except Exception as e:
        return {"error": f"Error extracting model info: {str(e)}"}

def process_joblib(file_path):
    """Carga un archivo joblib y obtiene su información relevante."""
    try:
        model = joblib.load(file_path)
        model_info = get_model_info(model)
        model_info["file_path"] = file_path  # Agregar la ruta del modelo para su uso en la UI
        return model_info
    except Exception as e:
        return {"error": f"Error loading model: {str(e)}"}

def start_conversation(model_info, model_description):
    """Orquesta la conversación entre los agentes y guarda el código generado."""
    llm_config = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key": os.getenv("DEEPINFRA_KEY"),
        "temperature": 0.7,
        "seed": 42,
    }

    model_analyzer = ConversableAgent(
        name="Model_Analyzer",
        llm_config=llm_config,
        system_message="You analyze machine learning models stored in joblib files and provide a structured report.",
        description="Analyzes ML models and provides a detailed report.",
    )

    ui_designer = ConversableAgent(
        name="UI_Designer",
        llm_config=llm_config,
        system_message=f"You design a Streamlit UI that effectively visualizes and interacts with the given model. The model is used for {model_description}. Ensure the UI has input fields corresponding to the expected number of features: {model_info['num_features']}",
        description="Creates a UI design for the model.",
    )

    code_generator = ConversableAgent(
        name="Code_Generator",
        llm_config=llm_config,
        system_message=f"""
        You implement a functional Streamlit app based on the provided UI design.
        The app is for {model_description} and should use the model located at '{model_info['file_path']}' for predictions.
        The UI should include {model_info['num_features']} input fields for user interaction.
        Ensure the model is loaded using joblib before making predictions.
        Generate only the Python code, without additional explanations, headers, or triple quotes.""",
        description="Generates the Streamlit application code.",
    )

    user_proxy = ConversableAgent(
        name="User_Proxy",
        description="Manages the workflow between agents.",
        llm_config=llm_config,
    )

    chat_results = []

    chat_results.append(user_proxy.initiate_chats([
        {"recipient": model_analyzer,
         "message": f"Analyze this machine learning model: {model_info}",
         "max_turns": 1,
         "summary_method": "last_msg"}
    ])[0])

    chat_results.append(user_proxy.initiate_chats([
        {"recipient": ui_designer,
         "message": "Design a Streamlit UI based on the analysis.",
         "max_turns": 1,
         "summary_method": "last_msg"}
    ])[0])

    chat_results.append(user_proxy.initiate_chats([
        {"recipient": code_generator,
         "message": "Generate the Streamlit application code based on the UI design.",
         "max_turns": 1,
         "summary_method": "last_msg"}
    ])[0])

    generated_code = chat_results[2].summary  # Obtener código generado
    
    # Guardar código en un archivo
    with open("generated_interface.py", "w", encoding="utf-8") as f:
        f.write(generated_code)
    
    return generated_code  # Retornar el código para ser usado en Streamlit

def main():
    file_path = input("Ingrese la ruta del archivo joblib: ").strip()

    if not os.path.isfile(file_path):
        print("Error: El archivo no existe. Verifique la ruta e intente nuevamente.")
        return

    model_info = process_joblib(file_path)
    if "error" in model_info:
        print(model_info["error"])
        return
    
    model_description = input("Describa brevemente para qué sirve este modelo: ").strip()
    
    start_conversation(model_info, model_description)

if __name__ == "__main__":
    main()
