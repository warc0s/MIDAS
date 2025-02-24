import torch
from transformers import BertForSequenceClassification, BertTokenizer
import re
import unicodedata

# Función para limpiar el texto
def clean_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Cargar el modelo y el tokenizador desde la carpeta "prompt_analysis"
model = BertForSequenceClassification.from_pretrained("prompt_analysis")
tokenizer = BertTokenizer.from_pretrained("prompt_analysis")

# Configurar el dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Función para clasificar la dificultad de un prompt
def clasificar_dificultad(texto):
    # Preprocesar el texto
    texto_limpio = clean_text(texto)
    # Tokenizar
    inputs = tokenizer(texto_limpio, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Realizar la predicción
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediccion = torch.argmax(logits, dim=1).item()
    return prediccion

# Bucle interactivo para recibir inputs del usuario
if __name__ == "__main__":
    while True:
        user_input = input("Ingrese su pregunta (o 'salir' para terminar): ")
        if user_input.lower() == 'salir':
            break
        dificultad = clasificar_dificultad(user_input)
        print(f"Dificultad clasificada: {dificultad}")