import joblib
import os
import json
import re
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from autogen import ConversableAgent
from dotenv import load_dotenv

load_dotenv()

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

        # Extraer nombres de features correctamente
        if hasattr(last_step, "feature_names_in_"):
            model_info["features"] = list(last_step.feature_names_in_)
        elif hasattr(last_step, "n_features_in_"):
            model_info["features"] = [f"feature_{i}" for i in range(last_step.n_features_in_)]
        else:
            model_info["features"] = []

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
        model_info["file_path"] = file_path
        return model_info
    except Exception as e:
        return {"error": f"Error loading model: {str(e)}"}

def load_json(json_path):
    """Carga un archivo JSON con información sobre features y la columna objetivo."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        features = json_data.get("features", [])
        target_column = json_data.get("target_column", None)

        if not features:
            return {"error": "El JSON no contiene una lista de features válida."}

        return {"features": features, "target_column": target_column}
    except json.JSONDecodeError:
        return {"error": "Error al leer el archivo JSON. Verifica su formato."}
    except Exception as e:
        return {"error": f"Error cargando el JSON: {str(e)}"}


def start_conversation(model_info, user_preferences):
    """Orquesta la conversación entre los agentes y genera el código de UI."""
    llm_config = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key": os.getenv("DEEPINFRA_KEY"),
        "temperature": 0.7,
        "seed": 42,
    }

    features_list = model_info.get("features", [])
    target_column = model_info.get("target_column", "Unknown")
    
    if isinstance(features_list, list):
        features_text = ", ".join(features_list)
    else:
        features_text = "Unknown"

    model_analyzer = ConversableAgent(
        name="Model_Analyzer",
        llm_config=llm_config,
        system_message="You analyze machine learning models stored in joblib files and provide a structured report.",
        description="Analyzes ML models and provides a detailed report.",
    )

    ui_designer = ConversableAgent(
        name="UI_Designer",
        llm_config=llm_config,
        system_message=f"""
        You design a Streamlit UI that effectively visualizes and interacts with the given model.
        Take into account the user's requirements, especially these preferences: {user_preferences}.
        The UI should include input fields corresponding to the following features: {features_text}.
        The model predicts the value of '{target_column}'.
        """,
        description="Creates a UI design for the model.",
    )

    code_generator = ConversableAgent(
        name="Code_Generator",
        llm_config=llm_config,
        system_message=f"""
        You implement a functional Streamlit app based on the provided UI design.
        The app should use the model located at '{model_info['file_path']}' for predictions.
        The UI should reflect the user's preferences: {user_preferences}.
        The UI should include input fields for the following features: {features_text}.
        The model's output should be displayed as the predicted '{target_column}' value.
        Ensure the model is loaded using joblib before making predictions.
        Generate only the Python code, without additional explanations, headers, or triple quotes.
        """,
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

    # Obtener código generado y limpiarlo directamente
    generated_code = re.sub(r"^```python\s*|\s*```$", "", chat_results[2].summary.strip())

    # Guardar código en un archivo
    with open("generated_interface.py", "w", encoding="utf-8") as f:
        f.write(generated_code)

    return generated_code  # Retornar el código para ser usado en Streamlit


def adjust_input_data(input_data, expected_features):
    """Asegura que los datos de entrada tengan las mismas columnas que el modelo espera."""
    df = pd.DataFrame([input_data])

    # Mantener solo las columnas esperadas y agregar las faltantes con 0
    df = df.reindex(columns=expected_features, fill_value=0)

    return df


def main():
    file_path = input("Ingrese la ruta del archivo joblib: ").strip()
    if not os.path.isfile(file_path):
        print("Error: El archivo no existe. Verifique la ruta e intente nuevamente.")
        return

    json_path = input("Ingrese la ruta del archivo JSON con las features (opcional, presione Enter para omitir): ").strip()

    model_info = process_joblib(file_path)
    if "error" in model_info:
        print(model_info["error"])
        return

    if json_path:
        if os.path.isfile(json_path):
            json_data = load_json(json_path)
            if "error" in json_data:
                print(json_data["error"])
                return

            # Verificar si las features del JSON coinciden con las del modelo
            if set(json_data["features"]) != set(model_info["features"]):
                print("⚠️ Advertencia: Las features en el JSON no coinciden con las del modelo. Ajustando...")
            
            model_info.update(json_data)
        else:
            print("⚠️ Advertencia: No se encontró el archivo JSON. Continuando sin él.")

    user_preferences = input("Indique colores o preferencias para la interfaz (opcional): ").strip()

    start_conversation(model_info, user_preferences)

if __name__ == "__main__":
    main()
