from google import genai
from google.genai import types
import base64
import os
from dotenv import load_dotenv

def generate():
    # Carga las variables de entorno desde el archivo .env
    load_dotenv()
    project = os.getenv("PROJECT")
    if project is None:
        print("No se encontró la variable 'PROJECT' en el archivo .env.")
        return

    # Solicita el prompt al usuario
    prompt = input("Introduce tu prompt: ")

    # Inicializa el cliente usando el proyecto obtenido del .env
    client = genai.Client(
        vertexai=True,
        project=project,
        location="us-central1",
    )

    model = "gemini-2.0-flash-001"
    # Se asigna el prompt ingresado a la lista de contenidos
    contents = [prompt]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
    )

    # Genera el contenido utilizando el modelo y muestra la respuesta por pantalla
    print("\nGenerando contenido...\n")
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
