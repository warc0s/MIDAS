from autogen import ConversableAgent
import pandas as pd
import os
from faker import Faker

import difflib

def detect_column_type(column_name):
    faker = Faker()
    column_name_lower = column_name.strip().lower()
    
    column_mapping = {
        "nombre": "name",
        "apellido": "last_name",
        "ciudad": "city",
        "direccion": "street_address",
        "estado": "state",
        "pais": "country",
        "codigo_postal": "postcode",
        "correo": "email",
        "telefono": "phone_number",
        "empresa": "company",
        "trabajo": "job",
        "descripcion": "text",
        "comentario": "sentence",
        "texto": "text",
        "fecha": "date",
        "cumpleaños": "date_of_birth",
        "creado": "date_time",
        "actualizado": "date_time",
        "tiempo": "time",
        "id": "uuid4",
        "numero": "random_int",
        "precio": "pyfloat",
        "cantidad": "pyfloat",
        "usuario": "user_name",
        "contraseña": "password",
        "url": "url",
        "dominio": "domain_name",
        "ip": "ipv4",
        "edad": "random_int",
        "dni": "random_int"
    }
    
    # Intentar encontrar una coincidencia exacta
    if column_name_lower in column_mapping:
        return column_mapping[column_name_lower]
    
    # Buscar una coincidencia cercana
    closest_match = difflib.get_close_matches(column_name_lower, column_mapping.keys(), n=1, cutoff=0.8)
    
    if closest_match:
        return column_mapping[closest_match[0]]

    return "text"  # Valor por defecto si no se encuentra una coincidencia

def generate_synthetic_data(num_records=100, columns=None):
    """Genera un dataset sintético basado en los parámetros especificados."""
    faker = Faker('es_ES')
    data = []
    
    column_types = {col: detect_column_type(col) for col in columns}
    print("Tipos de columnas detectados:", column_types)
    
    for _ in range(num_records):
        row = {}
        for col, faker_method in column_types.items():
            if faker_method == "random_int":
                row[col] = faker.random_int(min=18, max=99)
            elif hasattr(faker, faker_method):
                row[col] = getattr(faker, faker_method)()
            else:
                row[col] = faker.text()
        data.append(row)
    
    df = pd.DataFrame(data)
    print("Dataset generado:")
    print(df.head())  # Mostrar primeras filas para depuración
    return df

def start_conversation(user_request):
    """Orquesta la conversación entre los agentes para generar los datos."""
    llm_config = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key": os.getenv("DEEPINFRA_KEY"),
        "temperature": 0.7,
        "seed": 42,
    }

    input_agent = ConversableAgent(
        name="Input_Agent",
        llm_config=llm_config,
        system_message="You collect and validate user requirements for synthetic data generation. Ensure the response is clear and structured, without additional explanations.",
        description="Receives user input and ensures parameters are well-defined.",
    )
    
    validation_agent = ConversableAgent(
        name="Validation_Agent",
        llm_config=llm_config,
        system_message="You validate the dataset parameters provided by the user. Only confirm if they are valid or not, without asking for additional details.",
        description="Validates user input before dataset creation.",
    )
    
    column_classifier_agent = ConversableAgent(
        name="Column_Classifier_Agent",
        llm_config=llm_config,
        system_message="You classify column names to match corresponding Faker attributes for synthetic data generation.",
        description="Classifies column names before dataset generation.",
    )
    
    user_proxy = ConversableAgent(
        name="User_Proxy",
        description="Manages the workflow between agents.",
        llm_config=llm_config,
    )
    
    chat_results = []
    
    chat_results.append(user_proxy.initiate_chats([
        {"recipient": input_agent,
         "message": f"User requested dataset generation with parameters: {user_request}",
         "max_turns": 1,
         "summary_method": "last_msg"}
    ])[0])
    
    chat_results.append(user_proxy.initiate_chats([
        {"recipient": validation_agent,
         "message": "Validate the dataset parameters.",
         "max_turns": 1,
         "summary_method": "last_msg"}
    ])[0])
    
    chat_results.append(user_proxy.initiate_chats([
        {"recipient": column_classifier_agent,
         "message": "Classify the columns based on their names.",
         "max_turns": 1,
         "summary_method": "last_msg"}
    ])[0])
    
    dataset = generate_synthetic_data(user_request['num_records'], user_request['columns'])
    file_path = "synthetic_data.csv"
    
    try:
        dataset.to_csv(file_path, index=False)
        print(f"Dataset guardado en {file_path}")
    except Exception as e:
        print(f"Error al guardar el dataset: {e}")
    
    return dataset

def main():
    num_records = int(input("Número de registros: ").strip())
    columns = input("Nombres de las columnas (separadas por comas): ").strip().split(',')
    
    user_request = {
        "num_records": num_records,
        "columns": [col.strip() for col in columns]
    }
    
    dataset = start_conversation(user_request)

if __name__ == "__main__":
    main()
