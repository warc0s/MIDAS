from autogen import ConversableAgent
import pandas as pd
import os
from faker import Faker
import difflib

def detect_column_type(column_name):
    faker = Faker()
    column_name_lower = column_name.strip().lower()
    
    column_mapping = {
    # Personal Information
    "nombre": "name",
    "primer_nombre": "first_name",
    "segundo_nombre": "first_name",
    "apellido": "last_name",
    "apellido_paterno": "last_name",
    "apellido_materno": "last_name",
    "nombre_completo": "name",
    "genero": "random_element",
    "sexo": "random_element",
    "edad": "random_int",
    "fecha_nacimiento": "date_of_birth",
    "dni": "random_int",
    "cedula": "random_int",
    "pasaporte": "random_int",
    "curp": "random_int",
    "rfc": "random_int",

    # Contact Information
    "correo": "email",
    "email": "email",
    "telefono": "phone_number",
    "celular": "phone_number",
    "movil": "phone_number",
    "whatsapp": "phone_number",
    "red_social": "user_name",
    "usuario": "user_name",
    "nickname": "user_name",
    "contraseña": "password",
    "password": "password",

    # Address Information
    "direccion": "street_address",
    "calle": "street_name",
    "numero_exterior": "building_number",
    "numero_interior": "building_number",
    "colonia": "city",
    "municipio": "city",
    "ciudad": "city",
    "estado": "state",
    "region": "state",
    "pais": "country",
    "codigo_postal": "postcode",
    "zip": "postcode",

    # Company & Job Information
    "empresa": "company",
    "compania": "company",
    "negocio": "company",
    "puesto": "job",
    "cargo": "job",
    "departamento": "word",
    "sueldo": "pyfloat",
    "salario": "pyfloat",

    # Financial Information
    "precio": "pyfloat",
    "costo": "pyfloat",
    "descuento": "pyfloat",
    "cantidad": "pyfloat",
    "total": "pyfloat",
    "ingreso": "pyfloat",
    "gasto": "pyfloat",
    "deuda": "pyfloat",
    "credito": "pyfloat",
    "porcentaje": "pyfloat",
    "tasa": "pyfloat",

    # Time Information
    "fecha": "date",
    "fecha_nacimiento": "date_of_birth",
    "fecha_registro": "date_time",
    "fecha_creacion": "date_time",
    "fecha_modificacion": "date_time",
    "fecha_actualizacion": "date_time",
    "hora": "time",
    "tiempo": "time",
    "mes": "month",
    "año": "year",
    "semana": "random_int",
    "dia": "day_of_week",

    # Unique Identifiers
    "id": "uuid4",
    "identificador": "uuid4",
    "folio": "uuid4",
    "referencia": "uuid4",
    "codigo": "uuid4",
    "hash": "uuid4",

    # Web & Tech Information
    "ip": "ipv4",
    "ipv6": "ipv6",
    "mac": "mac_address",
    "url": "url",
    "dominio": "domain_name",
    "navegador": "user_agent",
    "sistema_operativo": "word",

    # Text & Descriptions
    "descripcion": "text",
    "comentario": "sentence",
    "notas": "sentence",
    "mensaje": "text",
    "resumen": "paragraph",
    "detalle": "text",
    "observaciones": "sentence",

    # Miscellaneous
    "color": "color_name",
    "emoji": "emoji",
    "serie": "uuid4",
    "numero": "random_int",
    "valor": "pyfloat",
    "cantidad_articulos": "random_int",
    "probabilidad": "pyfloat",
    "ranking": "random_int",
    "puntuacion": "random_int",
    "nivel": "random_int",
    "factor": "pyfloat"
}
    
    # 1. Intentar encontrar una coincidencia exacta
    if column_name_lower in column_mapping:
        return column_mapping[column_name_lower]

    # 2. Si la columna tiene más de una palabra, buscar coincidencias dentro del nombre
    for keyword, faker_type in column_mapping.items():
        if keyword in column_name_lower:
            return faker_type

    # 3. Buscar una coincidencia cercana con difflib
    closest_match = difflib.get_close_matches(column_name_lower, column_mapping.keys(), n=1, cutoff=0.8)
    if closest_match:
        return column_mapping[closest_match[0]]

    # 4. Si no se encuentra ninguna coincidencia, devolver "text" por defecto
    return "text"


def generate_synthetic_data(num_records=100, columns=None, constraints=None):
    """Genera un dataset sintético respetando los límites de valores numéricos si se especifican."""
    faker = Faker('es_ES')
    data = []

    # Asegurar que constraints no sea None
    constraints = constraints or {}

    column_types = {col: detect_column_type(col) for col in columns}
    print("Tipos de columnas detectados:", column_types)

    for _ in range(num_records):
        row = {}
        for col, faker_method in column_types.items():
            if faker_method == "random_int":
                min_val, max_val = constraints.get(col, (18, 99))  # Evitar error con get()
                row[col] = faker.random_int(min=min_val, max=max_val)
            elif faker_method == "pyfloat":
                min_val, max_val = constraints.get(col, (0.0, 100.0))
                row[col] = faker.pyfloat(left_digits=2, right_digits=2, min_value=min_val, max_value=max_val)
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
     system_message="""You classify column names to match Faker attributes for synthetic data generation.  
    Return a Python dictionary where the column name is the key and the Faker attribute is the value.  

    Classification Rules:  
    1. If the column name exactly matches a keyword in the predefined list, use that classification.  
    2. If the column name contains a relevant keyword (e.g., 'name', 'age', 'percentage'), assign the corresponding Faker attribute.  
    3. If the column name suggests a numerical value (e.g., 'percentage', 'ratio', 'rate', 'proportion'), classify it as 'pyfloat'.  
    4. If no match is found, classify it as 'text'.  

    Examples:  
    - 'employee_name' → 'name'  
    - 'client_lastname' → 'last_name'  
    - 'personal_email' → 'email'  
    - 'discount_percentage' → 'pyfloat'  
    - 'user_ID' → 'uuid4'  
    - 'customer_comment' → 'text'  
    - 'value_ratio' → 'pyfloat'  

    DO NOT PROVIDE PYTHON CODE.
    Only return a JSON dictionary.  
    """,
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
    
    dataset = generate_synthetic_data(
        num_records=user_request['num_records'], 
        columns=user_request['columns'], 
        constraints=user_request.get('constraints', {})
    )
    
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
    
    while True:
        print("\nOpciones:")
        print("1. Eliminar columna")
        print("2. Añadir columna")
        print("3. Finalizar y guardar")
        
        choice = input("Seleccione una opción: ").strip()
        
        if choice == "1":
            column_to_drop = input("Nombre de la columna a eliminar: ").strip()
            if column_to_drop in dataset.columns:
                dataset.drop(column_to_drop, axis=1, inplace=True)
                print(f"Columna '{column_to_drop}' eliminada.")
            else:
                print(f"La columna '{column_to_drop}' no existe.")
        elif choice == "2":
            new_column_name = input("Nombre de la nueva columna: ").strip()
            new_column_type = detect_column_type(new_column_name)
            faker = Faker('es_ES')
            
            if new_column_type == "random_int":
                dataset[new_column_name] = [faker.random_int(min=18, max=99) for _ in range(len(dataset))]
            elif hasattr(faker, new_column_type):
                dataset[new_column_name] = [getattr(faker, new_column_type)() for _ in range(len(dataset))]
            else:
                dataset[new_column_name] = [faker.text() for _ in range(len(dataset))]
            print(f"Columna '{new_column_name}' añadida.")
        elif choice == "3":
            file_path = "synthetic_data_modified.csv"
            try:
                dataset.to_csv(file_path, index=False)
                print(f"Dataset modificado guardado en {file_path}")
            except Exception as e:
                print(f"Error al guardar el dataset: {e}")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()
