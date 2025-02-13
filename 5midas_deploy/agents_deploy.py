import autogen
import joblib
import os
from sklearn.pipeline import Pipeline

config_list = [
    {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key": os.getenv("DEEPINFRA_KEY"),

    }
]

llm_config = {
    "config_list": config_list,
    "seed": 42,
    "temperature": 0.7
}

def create_agent(name, system_message):
    return autogen.AssistantAgent(
        name=name,
        llm_config=llm_config,
        system_message=system_message
    )

model_analyzer = create_agent(
    "model_analyzer",
    """You are an expert in analyzing machine learning models stored in joblib files. 
    Your task is to analyze the model and provide a clear, concise summary.
    When you finish your analysis, explicitly say 'ANALYSIS COMPLETE' and ask the UI Designer to proceed."""
)

ui_designer = create_agent(
    "ui_designer",
    """You are a UI/UX expert in Streamlit. Your task is to design a user interface based on the model analysis.
    When you finish your design, explicitly say 'UI DESIGN COMPLETE' and ask the Code Generator to proceed."""
)

code_generator = create_agent(
    "code_generator",
    """You are a Python and Streamlit developer. Your task is to implement the UI design into a working application.
    When you finish generating the code, explicitly say 'CODE GENERATION COMPLETE'."""
)

# Configuración mínima del GroupChat
groupchat = autogen.GroupChat(
    agents=[model_analyzer, ui_designer, code_generator],
    messages=[]
)

group_manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

class CustomUserProxyAgent(autogen.UserProxyAgent):
    def handle_message(self, message):
        """Monitorea los mensajes y detiene la conversación si se detecta 'CODE GENERATION COMPLETE'."""
        print(f"\n🔹 {message['name']}: {message['content']}")  # Muestra los mensajes en la terminal

        if "CODE GENERATION COMPLETE" in message["content"]:
            print("\n✅ Finalizando la conversación: Se ha completado la generación de código.")
            self.stop_all_agents()  # Detiene la ejecución de los agentes

user_proxy = CustomUserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

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

def start_conversation(model_info):
    initial_message = f"""
    Let's work together to create a Streamlit app for this ML model. Here's the model information:
    {model_info}
    
    Follow these steps in order:
    1. Model Analyzer: Please analyze the model structure and requirements.
    2. UI Designer: Based on the analysis, design the Streamlit interface.
    3. Code Generator: Create the final Streamlit application.
    
    Model Analyzer, please begin.
    """
    
    user_proxy.initiate_chat(
        group_manager,
        message=initial_message
    )

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
