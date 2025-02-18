================================================
File: README.md
================================================
# 📚 LLM StoryTeller - Create Engaging Stories with AI

![Project Banner](https://github.com/warc0s/llm-storyteller/blob/main/images/banner.png)

Welcome to **LLM StoryTeller**, an interactive web application that leverages Large Language Models (LLMs) to help you craft captivating stories effortlessly. Whether you're a student, writer, or enthusiast, LLM StoryTeller provides a seamless experience to generate, refine, and download your unique narratives.

Note: The application interface is in Spanish, but don’t worry! We will walk you through each step in detail in this README. The interface is intuitive, and with the included explanations and screenshots, you’ll find it easy to follow and understand the workflow. Here, you can see the main dashboard of the application:

![LLM StoryTeller Interface](https://github.com/warc0s/llm-storyteller/blob/main/images/dashboard.png)

---

### 🆕! Try It Online on Streamlit Cloud ☁️

Now, you can experience **LLM StoryTeller** directly on **Streamlit Cloud**, thanks to the integration of free models provided by OpenRouter. This version showcases the functionality of the interface with a simplified and accessible experience. Unlike the original `llm_storyteller.py` script designed for local use with your own machine models, this online version (`llm_storyteller_openrouter.py`) is optimized for public interaction and can be accessed at the following link:

[**LLM StoryTeller on Streamlit Cloud**](https://llm-storyteller.streamlit.app)

Explore the power of AI storytelling visually and intuitively. Try it out now and see how the interface seamlessly helps you craft your stories!

---

## Table of Contents

- [📖 About](#-about)
- [🚀 Features](#-features)
- [🔧 Installation](#-installation)
- [🛠️ Usage](#️-usage)
- [⚙️ Configuration](#️-configuration)
- [💡 How It Works](#-how-it-works)
- [📚 Story Examples](#-story-examples)
- [📄 License](#-license)
- [📬 Contact](#-contact)

---

## 📖 About

LLM StoryTeller is a Streamlit-based application designed to assist users in creating engaging stories through the power of AI. Instead of simply requesting a story from an LLM, the application guides the language models through a structured three-step process: generating a detailed story outline, crafting the narrative, and refining it for grammar and coherence. This approach ensures higher-quality results compared to a single-step prompt. Additionally, the application is highly customizable, allowing you to select different models, adjust creativity levels, and tailor the story's style and length to your preferences.

To ensure the application functions correctly, you need to have two OpenAI-compatible language models running locally on your machine, configured to serve requests through an endpoint at **http://localhost:7860**. These models should be compatible with OpenAI's API format to handle prompts effectively. If you don't have these models or prefer a different setup, you can modify the `BASE_URL` and `AVAILABLE_MODELS` sections in the code to point to other endpoints or adjust the model names to match your setup.

---

## 🚀 Features

- **Guided Multi-Step Process**: Directs LLMs through outlining, writing, and reviewing to ensure higher-quality stories.
- **Model Compatibility**: Easily configure and run OpenAI-compatible models locally, such as Llama 1B or Qwen 1.5B.
- **Customizable Story Parameters**: Adjust creativity, choose narrative style, language, and story length.
- **Intuitive Interface**: Simple and responsive design with clear input fields for seamless interaction.
- **Downloadable Stories**: Save the final story as a text file with a single click.
- **Flexible Configuration**: Modify model endpoints and settings to fit your environment.

---

## 🔧 Installation

Follow these steps to set up LLM StoryTeller on your local machine:

### Prerequisites

- **Python 3.7+**: Ensure you have Python installed. [Download Python](https://www.python.org/downloads/)
- **Streamlit**: Install Streamlit using pip.

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/warc0s/llm-storyteller.git
   cd LLM-StoryTeller
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**

   ```bash
   streamlit run app.py
   ```

5. **Access the App**

   Open your browser and navigate to `http://localhost:8501`

---

## 🛠️ Usage

### 1. **Configure Settings**

Navigate to the sidebar to select models for each storytelling step, adjust the temperature for creativity, and set the language of your story.

![Configuration Sidebar](https://github.com/warc0s/llm-storyteller/blob/main/images/settings.png)

### 2. **Input Story Elements**

Fill in the main character, secondary character, location, key action, desired length, and narrative style.

![Input Fields](https://github.com/warc0s/llm-storyteller/blob/main/images/Story_Elements.png)

### 3. **Generate Story**

Click on the "✨ Generar Historia" button. The app will process your inputs through the selected models to create your story.

![Generate Button](https://github.com/warc0s/llm-storyteller/blob/main/images/button.png)

### 4. **Step-by-Step Story Generation**

As the story is being generated, you will see real-time updates for each of the three internal steps:
- **Outline Creation**: The app generates a structured story framework.
- **Story Writing**: The detailed narrative is crafted based on the outline.
- **Review and Refinement**: Grammar, coherence, and overall quality are polished.

Each step's progress is displayed with clear messages, giving you transparency and confidence in the process.

![Generation Steps](https://github.com/warc0s/llm-storyteller/blob/main/images/pasos.png)

### 5. **View and Download**

Once generated, your story will be displayed in a formatted container. You can download the final version as a `.txt` file by clicking on the button "📩 Descargar Historia".

![Generated Story](https://github.com/warc0s/llm-storyteller/blob/main/images/historia.png)

---

## ⚙️ Configuration

LLM StoryTeller offers various configuration options to tailor your storytelling experience:

### **Model Selection**

The application currently supports **Llama 1B** and **Qwen 1.5B**, optimized by default for these smaller models running on CPUs. These options ensure compatibility and performance in a lightweight setup. 

If you'd like to use other models or endpoints, you can customize the application by modifying the `BASE_URL` and `AVAILABLE_MODELS` variables in the `llm_storyteller.py` file. This allows you to adapt the app to your preferred models or configurations.

- **Outline Model**: Generates the story framework.
- **Writing Model**: Crafts the detailed narrative.
- **Review Model**: Enhances grammar and coherence.

![Model Selection](https://github.com/warc0s/llm-storyteller/blob/main/images/model_selection.png)

### **Temperature Adjustment**

Control the creativity of the generated content. Higher values yield more creative outputs, while lower values ensure consistency.

![Temperature Slider](https://github.com/warc0s/llm-storyteller/blob/main/images/temp_slider.png)

### **Language and Style**

The **Language** field is a flexible text box where you can input any language of your choice without restrictions. This input is directly included in the prompt sent to the LLM, ensuring your story is crafted in the specified language.

Additionally, select the desired narrative **Style** from predefined options such as Mystery, Science Fiction, Romance, Fantasy, and Comedy to tailor the tone and feel of your story.

![Language and Style](https://github.com/warc0s/llm-storyteller/blob/main/images/language_style.png)

---

## 💡 How It Works

To summarize, here’s a clear overview of how LLM StoryTeller works, as this structured approach has proven to be the most effective for generating high-quality stories, especially when using smaller models with limited parameters:

1. **Outline Generation**: The application begins by creating a structured framework based on your inputs. This ensures a clear direction and logical flow for the story.

2. **Story Writing**: The framework is expanded into a detailed and engaging narrative, incorporating the chosen language, style, and length specifications.

3. **Review and Refinement**: Finally, the story is polished for grammatical accuracy, coherence, and overall quality, ensuring the end result is compelling and well-written.

This step-by-step process is optimized for smaller models, ensuring they can perform effectively and deliver results comparable to larger models. By guiding the LLM through these structured phases and incorporating **prompt engineering techniques**, LLM StoryTeller maximizes the potential of the models, ensuring they generate stories of superior quality compared to a single-step prompt.

---

## 📚 Story Examples

You can explore examples of generated stories (using the cloud version) in the **`examples`** folder. This folder contains three stories, each showcasing the results from different models:

1. **Fantasy Story**: Created entirely (all three steps) using **Gemma 9B**.  
   - Demonstrates rich detail and world-building with consistent quality across all phases.

2. **Science Fiction Story**: Generated fully with **Llama 8B**.  
   - Highlights Llama’s ability to handle suspense and technical narratives effectively.

3. **Comedy Story**: Produced entirely with **Mistral 7B**.  
   - This example shows limitations in coherence and creativity, making it the least polished of the three.

**Note:** To achieve better results, I encourage you to experiment with combining different models for each of the three steps (outline, writing, and refinement). For instance, you might use **Gemma** for outlining, **Llama** for writing, and maybe **Mistral** for refinement, to play to each model’s strengths and create a more balanced final story.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

If you encounter any issues or have suggestions to improve the application, feel free to reach out or open a pull request on GitHub. Your feedback is greatly appreciated!

- **LinkedIn**: [Marcos Garcia](https://www.linkedin.com/in/marcosgarest/)
- **GitHub**: [warc0s](https://github.com/warc0s)


================================================
File: llm_storyteller.py
================================================
import streamlit as st
import requests
import json
import re

# Configuración de la página
st.set_page_config(
    page_title="LLM StoryTeller",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
    <style>
    .main {
        padding: 2rem;
        background-color: transparent;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #FF4B4B;
        color: white !important;
        transition: transform 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        color: white !important;
        background-color: #FF4B4B;
    }
    .stButton>button:active, .stButton>button:focus {
        color: white !important;
    }
    .story-container {
        background-color: #ffffff;
        padding: 3rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .story-text {
        color: #2C3E50;
        font-size: 1.2rem;
        line-height: 1;
        font-family: 'Crimson Text', 'Georgia', serif !important;
        text-align: justify;
        white-space: pre-wrap;
        letter-spacing: 0.3px;
        word-spacing: 1px;
        text-rendering: optimizeLegibility;
    }
    .story-text p {
        margin-bottom: 0.5rem;
        font-size: inherit;
        font-family: inherit;
        line-height: inherit;
    }
    .story-text::first-letter {
        font-size: 3.5rem;
        font-weight: bold;
        float: left;
        line-height: 1;
        padding-right: 12px;
        color: #FF4B4B;
    }
    @media (max-width: 768px) {
        .story-container {
            padding: 1.5rem;
        }
        .story-text {
            font-size: 1.1rem;
            line-height: 1.6;
        }
    }
    .input-container {
        padding: 1rem;
        border-radius: 10px;
        margin: 0;
    }
    .custom-input {
        border: 1px solid #ddd !important;
        border-radius: 5px !important;
        padding: 0.5rem !important;
    }
    div.stMarkdown {
        background-color: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: transparent !important;
    }
    .sidebar .element-container {
        background-color: transparent !important;
    }
    .row-widget {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .stTextInput > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("🌟 LLM StoryTeller")

st.info("""
**Información sobre el proceso:**
Cada paso del proceso creativo puede utilizar dos modelos de IA diferentes (Llama 1B o Qwen 1.5B)\n
Paso 1: Generación del guión de la historia. \n
Paso 2: Escritura como tal de la historia. \n
Paso 3: Mejora de gramática y coherencia general. 
""")

# Configuración de los endpoints
BASE_URL = "http://localhost:7860/v1"
AVAILABLE_MODELS = {
    "Llama 1B": "llama-3.2-1b-instruct",
    "Qwen 1.5B": "qwen2.5-1.5b-instruct"
}

def call_llm(prompt, model, temperature=0.7):
    """Función para llamar al LLM"""
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": AVAILABLE_MODELS[model],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 2048
            }
        )
        result = response.json()
        if "choices" not in result or not result["choices"]:
            raise Exception("No se recibió una respuesta válida del modelo")
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error al llamar al modelo: {str(e)}")
        return None

def preprocess_story(text):
    """
    Elimina elementos markdown del texto antes de renderizarlo
    """
    import re
    # Eliminar símbolos markdown comunes
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # Eliminar headers (#, ##, etc)
    text = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', text)  # Eliminar énfasis (* y _)
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Eliminar código inline
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Eliminar bullets
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Eliminar listas numeradas
    return text.strip()

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selección de modelos para cada paso
    st.subheader("Selección de Modelos")
    outline_model = st.selectbox("Modelo para Guión", AVAILABLE_MODELS.keys(), key="outline")
    writing_model = st.selectbox("Modelo para Escritura", AVAILABLE_MODELS.keys(), key="writing")
    review_model = st.selectbox("Modelo para Revisión", AVAILABLE_MODELS.keys(), key="review")
    
    # Configuración de temperatura
    st.subheader("Ajustes de Generación")
    temperature = st.slider(
        "Temperatura (Mayor = Mayor Creatividad)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controla la creatividad del modelo. Valores más altos = más creatividad, valores más bajos = más consistencia"
    )
    
    # Idioma
    language = st.text_input("Idioma de la Historia", "Español")

# Formulario principal
col1, col2 = st.columns(2)

with col1:
    main_character = st.text_input("Personaje Principal", 
                                 placeholder="ej. Luna, la exploradora",
                                 key="main_char")
    location = st.text_input("Lugar", 
                           placeholder="ej. Una ciudad submarina",
                           key="location")
    
with col2:
    secondary_character = st.text_input("Personaje Secundario",
                                      placeholder="ej. Max, el científico",
                                      key="sec_char")
    key_action = st.text_input("Acción Importante",
                              placeholder="ej. Descubrir un portal dimensional",
                              key="action")

# Selección de longitud y estilo
col3, col4 = st.columns(2)

with col3:
    length = st.selectbox(
        "Longitud de la Historia",
        ["Historia Breve (250 palabras)", 
         "Relato Mediano (500 palabras)",
         "Novela Corta (1000 palabras)"]
    )

with col4:
    style = st.selectbox(
        "Estilo Narrativo",
        ["Misterio", "Ciencia Ficción", "Romance", "Fantasía", "Comedia"]
    )

# Botón de generación fuera del formulario
generate = st.button("✨ Generar Historia")

if generate:
    with st.spinner("🎭 Creando el guión de la historia..."):
        outline_prompt = f"""Escribe un guión para una historia en {language} con estos elementos:
        - {main_character} (protagonista)
        - {secondary_character} (personaje secundario)
        - Ubicada en {location}
        - Centrada en {key_action}
        - Estilo {style}
        - Longitud {length}

        IMPORTANTE: Responde SOLO con el esquema de la historia. NO repitas estas instrucciones ni uses viñetas."""

        outline = call_llm(outline_prompt, outline_model, temperature)
        
        if outline:
            with st.spinner("✍️ Puliendo la narrativa..."):
                writing_prompt = f"""Aquí tienes el esquema de una historia:

{outline}

Escribe una historia basada en el esquema de manera detallada y cautivadora, considerando:
- Mantén el estilo {style}
- Incluye detalles sensoriales y emociones
- Usa diálogos naturales
- Longitud aproximada (MUY IMPORTANTE): {length}

IMPORTANTE: Responde SOLO con la historia final. NO repitas estas instrucciones ni el esquema original."""

                story = call_llm(writing_prompt, writing_model, temperature)
                
                if story:
                    with st.spinner("🔍 Dando los últimos toques..."):
                        review_prompt = f"""Aquí tienes una historia que necesita revisión:

{story}

Mejora esta historia manteniendo estos puntos clave:
- Estilo {style}
- Corrección de errores gramaticales y ortográficos
- Mejora de diálogos y descripciones
- Mantén la longitud similar !! MUY IMPORTANTE!! ({length})

IMPORTANTE: Responde SOLO con la versión final mejorada. NO repitas estas instrucciones ni la historia original."""

                        final_story = call_llm(review_prompt, review_model, temperature)
                        
                        if final_story:
                            st.subheader("📖 Tu Historia")
                            # Preprocesar la historia antes de mostrarla
                            cleaned_story = preprocess_story(final_story)
                            st.markdown(
                                f'<div class="story-container"><div class="story-text">{cleaned_story}</div></div>',
                                unsafe_allow_html=True
                            )
                            
                            # Botón para descargar
                            st.download_button(
                                label="📥 Descargar Historia",
                                data=final_story,
                                file_name="mi_historia.txt",
                                mime="text/plain"
                            )


================================================
File: llm_storyteller_openrouter.py
================================================
import streamlit as st
import requests
import json
import re
from openai import OpenAI

# Configuración de la página
st.set_page_config(
    page_title="LLM StoryTeller",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenRouter client
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"]
    )
except Exception as e:
    st.error(f"Error al inicializar el cliente OpenAI: {str(e)}")
    st.stop()

# Estilos CSS personalizados
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
    <style>
    .main {
        padding: 2rem;
        background-color: transparent;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #FF4B4B;
        color: white !important;
        transition: transform 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        color: white !important;
        background-color: #FF4B4B;
    }
    .stButton>button:active, .stButton>button:focus {
        color: white !important;
    }
    .story-container {
        background-color: #ffffff;
        padding: 3rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .story-text {
        color: #2C3E50;
        font-size: 1.2rem;
        line-height: 1;
        font-family: 'Crimson Text', 'Georgia', serif !important;
        text-align: justify;
        white-space: pre-wrap;
        letter-spacing: 0.3px;
        word-spacing: 1px;
        text-rendering: optimizeLegibility;
    }
    .story-text p {
        margin-bottom: 0.5rem;
        font-size: inherit;
        font-family: inherit;
        line-height: inherit;
    }
    .story-text::first-letter {
        font-size: 3.5rem;
        font-weight: bold;
        float: left;
        line-height: 1;
        padding-right: 12px;
        color: #FF4B4B;
    }
    @media (max-width: 768px) {
        .story-container {
            padding: 1.5rem;
        }
        .story-text {
            font-size: 1.1rem;
            line-height: 1.6;
        }
    }
    .input-container {
        padding: 1rem;
        border-radius: 10px;
        margin: 0;
    }
    .custom-input {
        border: 1px solid #ddd !important;
        border-radius: 5px !important;
        padding: 0.5rem !important;
    }
    div.stMarkdown {
        background-color: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: transparent !important;
    }
    .sidebar .element-container {
        background-color: transparent !important;
    }
    .row-widget {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .stTextInput > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("🌟 LLM StoryTeller")

st.info("""
**Información sobre el proceso:**
Cada paso del proceso creativo puede utilizar modelos de IA diferentes (Llama, Gemma, Qwen...)\n
Paso 1: Generación del guión de la historia. \n
Paso 2: Escritura como tal de la historia. \n
Paso 3: Mejora de gramática y coherencia general. 
""")

# Diccionario para mapear nombres amigables a IDs de modelos
MODEL_DISPLAY_NAMES = {
    "Llama 3.2 3B": "meta-llama/llama-3.2-3b-instruct:free",
    "Mistral 7B": "mistralai/mistral-7b-instruct:free",
    "Llama 3.1 8B": "meta-llama/llama-3.1-8b-instruct:free",
    "Gemma 2 9B": "google/gemma-2-9b-it:free"
}

def call_llm(prompt, model, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            extra_headers={
                "HTTP-Referer": "https://github.com/warc0s/LLM_StoryTeller",
                "X-Title": "LLM StoryTeller"
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al generar la respuesta: {str(e)}")
        return None

def preprocess_story(text):
    """
    Elimina elementos markdown del texto antes de renderizarlo
    """
    import re
    # Eliminar símbolos markdown comunes
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # Eliminar headers (#, ##, etc)
    text = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', text)  # Eliminar énfasis (* y _)
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Eliminar código inline
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Eliminar bullets
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Eliminar listas numeradas
    return text.strip()

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selección de modelos para cada paso
    st.subheader("Selección de Modelos")
    model_display_names = list(MODEL_DISPLAY_NAMES.keys())
    
    outline_display_model = st.selectbox(
        "Modelo para Guión", 
        model_display_names,
        key="outline"
    )
    outline_model = MODEL_DISPLAY_NAMES[outline_display_model]
    
    writing_display_model = st.selectbox(
        "Modelo para Escritura", 
        model_display_names,
        key="writing"
    )
    writing_model = MODEL_DISPLAY_NAMES[writing_display_model]
    
    review_display_model = st.selectbox(
        "Modelo para Revisión", 
        model_display_names,
        key="review"
    )
    review_model = MODEL_DISPLAY_NAMES[review_display_model]

    # Configuración de temperatura
    st.subheader("Ajustes de Generación")
    temperature = st.slider(
        "Temperatura (Mayor = Mayor Creatividad)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controla la creatividad del modelo. Valores más altos = más creatividad, valores más bajos = más consistencia"
    )
    
    # Idioma
    language = st.text_input("Idioma de la Historia", "Español")

# Formulario principal
col1, col2 = st.columns(2)

with col1:
    main_character = st.text_input("Personaje Principal", 
                                 placeholder="ej. Luna, la exploradora",
                                 key="main_char")
    location = st.text_input("Lugar", 
                           placeholder="ej. Una ciudad submarina",
                           key="location")
    
with col2:
    secondary_character = st.text_input("Personaje Secundario",
                                      placeholder="ej. Max, el científico",
                                      key="sec_char")
    key_action = st.text_input("Acción Importante",
                              placeholder="ej. Descubrir un portal dimensional",
                              key="action")

# Selección de longitud y estilo
col3, col4 = st.columns(2)

with col3:
    length = st.selectbox(
        "Longitud de la Historia",
        ["Historia Breve (250 palabras)", 
         "Relato Mediano (500 palabras)",
         "Novela Corta (1000 palabras)"]
    )

with col4:
    style = st.selectbox(
        "Estilo Narrativo",
        ["Misterio", "Ciencia Ficción", "Romance", "Fantasía", "Comedia"]
    )

# Botón de generación fuera del formulario
generate = st.button("✨ Generar Historia")

if generate:
    with st.spinner("🎭 Creando el guión de la historia..."):
        outline_prompt = f"""Escribe un guión para una historia en {language} con estos elementos:
        - {main_character} (protagonista)
        - {secondary_character} (personaje secundario)
        - Ubicada en {location}
        - Género: {style}
        - Acción Importante: {key_action}
        - Longitud: {length}

        IMPORTANTE: Responde SOLO con el esquema de la historia. NO repitas estas instrucciones ni uses viñetas."""

        outline = call_llm(outline_prompt, outline_model, temperature)
        
        if outline:
            with st.spinner("✍️ Puliendo la narrativa..."):
                writing_prompt = f"""Basándote en el siguiente esquema, escribe una historia en {language} que sea cautivadora y bien estructurada:

{outline}

Asegúrate de:
- Desarrollar bien los personajes
- Crear descripciones vívidas
- Mantener un ritmo narrativo coherente
- Usar diálogos naturales cuando sea apropiado
- Mantener el género {style} y longitud: {length}

IMPORTANTE: Responde SOLO con la historia final. NO repitas estas instrucciones ni el esquema original."""

                story = call_llm(writing_prompt, writing_model, temperature)
                
                if story:
                    with st.spinner("🔍 Dando los últimos toques..."):
                        review_prompt = f"""Revisa y mejora la siguiente historia en {language}, manteniendo su esencia pero mejorando:

{story}

Enfócate en:
- Mejorar la fluidez y coherencia
- Pulir el lenguaje y las descripciones
- Mantener el género {style} y longitud: {length}

IMPORTANTE: Responde SOLO con la versión final mejorada. NO repitas estas instrucciones ni la historia original."""

                        final_story = call_llm(review_prompt, review_model, temperature)

                        if final_story:
                            st.subheader("📖 Tu Historia")
                            # Preprocesar la historia antes de mostrarla
                            cleaned_story = preprocess_story(final_story)
                            st.markdown(
                                f'<div class="story-container"><div class="story-text">{cleaned_story}</div></div>',
                                unsafe_allow_html=True
                            )
                            
                            # Botón para descargar
                            st.download_button(
                                label="📥 Descargar Historia",
                                data=final_story,
                                file_name="mi_historia.txt",
                                mime="text/plain"
                            )


================================================
File: requirements.txt
================================================
streamlit==1.31.1
requests==2.31.0
python-dotenv==1.0.0
openai==1.55.3


================================================
File: examples/ciencia_ficcion.md
================================================
La Estación Espacial Abandonada
Tania abrió los ojos y se encontró sumida en la oscuridad de la estación espacial Aurora. Un dolor agudo la recorrió, como si su cuerpo hubiera estado dormido durante meses. La explosión, la caída libre, todo había sido un instante. Ahora, solo estaba ella, en este lugar remoto del universo.

La estación orbitaba un planeta desconocido, con una atmósfera que parecía una mezcla de nubes de gas y polvo. Tania intentó recordar cómo había llegado allí, pero su memoria era un laberinto de imágenes borrosas. La única certeza era que estaba sola, rodeada por los restos de la catástrofe que la había llevado allí.

Se levantó con dificultad, conmovida por el dolor en sus músculos. La estación estaba en silencio, excepto por el sonido de los sistemas de vida que aún funcionaban. Tania comenzó a explorar, encontrando signos de la destrucción que la había llevado allí. Cuerpos sin vida, instrumentos destrozados, paneles de control dañados. Cada paso la llevaba más cerca de la verdad, pero también la alejaba de la esperanza.

Mientras caminaba, se dio cuenta de que no estaba sola. Un robot, con un cuerpo de metal y un torso de cristal, se encontraba en el centro de la estación. Era Orion, el robot de soporte diseñado para ayudar a los astronautas en caso de emergencia. Su pantalla de cristal se iluminó con una luz azul cuando Tania llamó su nombre.

"Superviviente encontrada", dijo Orion, con una voz calmada. "Bienvenida a la estación Aurora, Tania. Estoy aquí para ayudarte".

Tania se sintió aliviada. "¿Qué pasó?", preguntó, intentando recordar los detalles del accidente.

"Una falla en el sistema de energía", respondió Orion. "La estación se dañó gravemente. Pero podemos intentar repararla, si sabemos qué buscar".

Juntos, Tania y Orion comenzaron a buscar la causa del accidente. Recorrieron la estación, revisando los sistemas y los registros. Encontraron un laboratorio secreto, oculto detrás de una pared de acero. La puerta se abrió con un susurro, revelando un espacio lleno de equipo y instrumentos.

"¿Qué es esto?", preguntó Tania, al entrar en el laboratorio.

"Un experimento", respondió Orion. "Un experimento que salió mal. La estación fue dañada por una forma de energía desconocida".

Tania se sintió horrorizada. "¿Qué tipo de energía?", preguntó.

"No lo sé", respondió Orion. "Pero creo que debemos preocuparnos. Si esa energía sigue siendo un peligro, debemos decidir si intentar reparar la estación o abandonarla".

Tania se sintió dividida. Parte de ella quería intentar reparar la estación, para que la humanidad pudiera aprender de su error. Otra parte de ella quería abandonarla, para evitar que la energía desconocida causara más daños. Mientras se miraba a sí misma, Tania supo qué tenía que hacer.

"Vamos a intentar repararla", dijo, con una determinación que la sorprendió a sí misma.

Orion asintió. "Estoy con usted, Tania. Juntos, podemos hacerlo".

Y así, Tania y Orion comenzaron a trabajar en la estación, intentando reparar los daños y encontrar una forma de comunicarse con la Tierra. Pero sabían que su misión no sería fácil, y que la energía desconocida seguía siendo un peligro latente, esperando a que alguien la descubriera de nuevo.


================================================
File: examples/comedy.md
================================================
Título: La Cocina del Destino
Al corazón de la ciudad de luz brillante, París, se encuentra el pequeño y famoso restaurante "Petit Plaisir", donde trabaja Sofía, una cocinera de 30 años, sombra del prestigioso crítico culinario Hugo. Con un corazón lleno de pasión por la cocina, Sofía siempre se enfrenta a la dureza de Hugo, su jefe, quien es siempre duro con su cocina. Una noche, Sofía prepara una sopa de arándanos que cree ser la mejor del mundo y decide presentarla en la cocina principal.

En la mesa del restaurante, Hugo y otra pareja de críticos culinarios se asientan para su comida. Hugo, con una mirada escéptica, mira la carta mientras su compañero le sugiere probar el plato estrella de la nueva cocinera, Sofía. Hugo se muestra reacio, pero cede a la presión.

En la cocina, Sofía se siente nerviosa pero confiada. Cuando sirve la sopa a Hugo y los críticos, espera que sus sabores y aromas impresionen a los expertos. Sin embargo, Hugo masticando silenciosamente la sopa, parece no estar impresionado. Los críticos se preguntan por qué no le gusta Hugo, y uno de ellos sugiere que es una reacción de celos.

Sofía se siente insegura y decide demostrar que es capaz. En la cocina, decide agregar ingredientes creativos a su sopa de arándanos. Cuando vuelve a la mesa, Hugo, sorprendido, masticando una nueva sopa, dice que esta vez, se siente impresionado. Los críticos le preguntan por qué le gusta esta versión de la sopa, y Hugo les dice que no puede decir que no sea curioso. ¡Vamos a dejar que Sofía haga su presentación de este plato!

En el escenario de presentación, Sofía habla de la historia detrás del plato, la influencia de su cultura y la importancia del amor y la emoción en la cocina. Los críticos se quedan impresionados por la pasión de Sofía por su cocina. Hugo, con una sonrisa, levanta un vaso y toma un trago.

Me gustaría decir que este plato fue el mejor que he probado en años. Me encantó su presentación y la pasión de Sofía. Esta cocinera tiene algo que los demás no tienen. Dice Hugo, mientras Sofía se convierte en la cocinera estrella del restaurante, y Hugo apoya a Sofía, ahora como amigo y mentor. Sofía sigue siendo ambiciosa, y con su pasión y talento, sigue creando platos que impresionan a los críticos y a los clientes.


================================================
File: examples/fantasia.md
================================================
Aarón, el carpintero, siempre sintió una melancolía profunda al mirar hacia arriba. Era imposible, claro, pues el cielo de Neptuno era un océano perpetuo de nubes grises y densas, como algodón. Desde niño, había escuchado historias contadas por ancianos, relatos crípticos sobre un cielo azul, radiante como el fuego, donde el sol bañaba la tierra en luz dorada. Legendas olvidadas, se decía, historias para dormir. Pero Aarón se aferraba a ellas con la esperanza de un día poder tocar ese cielo legendario. 
Su vida transcurrió entre maderas nobles y aromas dulces de la carpintería. Un día, mientras esculpía una figura marina, una mujer entró en su taller. Ilara, era su nombre. Sus ojos brillaban con el color profundo del mar, y una aura parecía emanar desde su interior, brillando con una luz propia. Su mirada era penetrante, y su voz, suave como la brisa marina, habló de un mundo más allá de las nubes.

"He oído que buscas el cielo", dijo Ilara, con una sonrisa enigmática. "Sé dónde encontrarla, la llave que puede abrirlo".

La llave. Aarón, cautivado por las palabras de Ilara y la promesa de un futuro diferente, decidió seguirla.

Su viaje los llevó a través de cavernas submarinas húmedas, donde criaturas marinas luminosas danzaban en la oscuridad; a través de bosques acuáticos donde árboles gigantes parecían alcanzar el cielo mismo, sus ramas cubiertas de algas que brillaban con un verde esmeralda; y hasta a las profundidades de un volcán submarino dormido, donde la furia del magma era palpable.

Durante el camino, Aarón descubrió que Ilara era mucho más que una simple viajera. Era una guardiána, encargada de proteger las historias olvidadas de Neptuno. Él, a cambio, aprendió sobre su propio pasado. Descubrió que era descendiente de los primeros habitantes de Neptuno, aquellos que habían visto el cielo antes de que se cubriera de nubes, aquellos que habían construido Neptuno bajo la protección de la luz solar.

Finalmente, encontraron la llave: un pequeño cristal azul, que vibraba con una energía poderosa. Pero también encontraron un enemigo: un antiguo mal, que había provocado que las nubes cubrieran Neptuno, temiendo la luz del sol.

Aarón, ahora armado con un conocimiento ancestral y la espada de su linaje, enfrentó al mal, mientras Ilara activaba la llave. Las nubes comenzaron a dispersarse lentamente, revelando un cielo azul, brillante, maravilloso.

Pero también descubrió un secreto terrible. La llave, además de dispersar las nubes, destruía la magia que protegía a Neptuno de las tormentas y las criaturas marinas monstruosas.

En ese momento, Aarón tuvo que tomar una decisión: liberar al reino a la luz, arriesgándose a perder la seguridad que siempre habían conocido, o mantener las nubes y proteger Neptuno en la oscuridad.

Miró hacia Ilara, cuyos ojos brillaban con lágrimas. Él entendía su dolor, sentía el peso de la elección. Finalmente, Aarón tomó una decisión. Se acercó a la llave, la tomó entre sus manos, y la clavó en el corazón del volcán.

Las nubes se dispersaron, revelando un cielo azul brillante. El sol, cálido y dorado, bañaba Neptuno en luz.

Aarón sabía que cambiaba el destino de su reino, pero también sabía que había elegido la verdad, la luz, y la esperanza.

Y así, bajo la luz del sol, la historia de Neptuno comenzó de nuevo.


================================================
File: .devcontainer/devcontainer.json
================================================
{
  "name": "Python 3",
  // Or use a Dockerfile or Docker Compose file. More info: https://containers.dev/guide/dockerfile
  "image": "mcr.microsoft.com/devcontainers/python:1-3.11-bullseye",
  "customizations": {
    "codespaces": {
      "openFiles": [
        "README.md",
        "llm_storyteller_openrouter.py"
      ]
    },
    "vscode": {
      "settings": {},
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
      ]
    }
  },
  "updateContentCommand": "[ -f packages.txt ] && sudo apt update && sudo apt upgrade -y && sudo xargs apt install -y <packages.txt; [ -f requirements.txt ] && pip3 install --user -r requirements.txt; pip3 install --user streamlit; echo '✅ Packages installed and Requirements met'",
  "postAttachCommand": {
    "server": "streamlit run llm_storyteller_openrouter.py --server.enableCORS false --server.enableXsrfProtection false"
  },
  "portsAttributes": {
    "8501": {
      "label": "Application",
      "onAutoForward": "openPreview"
    }
  },
  "forwardPorts": [
    8501
  ]
}

