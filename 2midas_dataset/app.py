import streamlit as st
import pandas as pd
import os
from io import StringIO, BytesIO
from dotenv import load_dotenv
from agents_dataset import start_conversation, detect_column_type, generate_synthetic_data

load_dotenv()

st.set_page_config(
    page_title="Midas Dataset | Midas System",
    page_icon="https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/dataset_trans.png?raw=true",
)

logo_url = "https://github.com/warc0s/MIDAS/blob/main/Extra/Logos/transparentes/dataset_trans.png?raw=true"

# 🎨 Estilos personalizados
st.markdown(
    f"""
    <style>
        body {{ background-color: #0A0F22; }}
        .title-container h1 {{ color: #DAA520; }}
        .title-container {{ display: flex; align-items: center; gap: 15px; }}
        .title-container img {{ width: 60px; height: 60px; }}

        /* Estilos para las tarjetas */
        .custom-box {{
            background-color: #1E1E2F;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(255, 215, 0, 0.2);
            margin-bottom: 15px;
        }}

        .custom-title {{
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 10px;
        }}

        .stNumberInput label, .stTextInput label, .stSelectbox label {{
            font-size: 23px !important;
            font-weight: bold;
        }}

        .download-button {{
            background-color: #DAA520;
            color: white;
            font-weight: bold;
            padding: 10px 15px;
            border-radius: 5px;
            margin-top: 15px;
            display: inline-block;
        }}
    </style>
    <div class="title-container">
        <img src="{logo_url}" alt="Logo">
        <h1>Midas Dataset</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# 1️⃣ Inputs para el dataset
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    num_records = st.number_input("Número de registros", min_value=1, value=100)
with col2:
    columns_input = st.text_input("Nombres de las columnas (separadas por comas)", "Nombre, Apellido, Edad")
columns = [col.strip() for col in columns_input.split(',')]

# ---------------------------
# 2️⃣ Capturar mínimos y máximos en la misma línea
# ---------------------------
if "constraints" not in st.session_state:
    st.session_state.constraints = {}

for col in columns:
    col_type = detect_column_type(col)
    if col_type in ["random_int", "pyfloat"]:  
        col_min_max = st.columns(2)
        with col_min_max[0]:
            min_val = st.number_input(f"Min {col}", value=0, key=f"min_{col}")
        with col_min_max[1]:
            max_val = st.number_input(f"Max {col}", value=100, key=f"max_{col}")
        st.session_state.constraints[col] = (min_val, max_val)
        

# ---------------------------
# 3️⃣ Generar datos
# ---------------------------
if st.button("Generar Datos", key="generar"):
    user_request = {
        "num_records": num_records,
        "columns": columns,
        "constraints": st.session_state.constraints
    }
    
    try:
        dataset = start_conversation(user_request)
        st.session_state.dataset = dataset  
        st.success("✅ Datos generados correctamente.")
        st.dataframe(dataset, use_container_width=True)  
    except Exception as e:
        st.error(f"❌ Error al generar los datos: {e}")

# ---------------------------
# 4️⃣ Modificar el dataset generado
# ---------------------------
if 'dataset' in st.session_state:
    st.markdown("---")
    st.header("Modificar Dataset")

    col1, col2 = st.columns(2)

    # 🗑️ Tarjeta para eliminar columnas
    with col1:
        with st.container():
            st.markdown('<p class="custom-title">🗑️ Eliminar Columna</p>', unsafe_allow_html=True)
            column_to_drop = st.selectbox("Selecciona la columna a eliminar", st.session_state.dataset.columns)
            if st.button("Eliminar", key="eliminar", help="Elimina la columna seleccionada"):
                try:
                    st.session_state.dataset.drop(column_to_drop, axis=1, inplace=True)
                    st.success(f"✅ Columna '{column_to_drop}' eliminada.")
                    st.dataframe(st.session_state.dataset, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error al eliminar la columna: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ➕ Tarjeta para añadir columnas
    with col2:
        with st.container():
            st.markdown('<p class="custom-title">➕ Añadir Columna</p>', unsafe_allow_html=True)

            new_column_name = st.text_input("Nombre de la nueva columna", key="añadir")

            if new_column_name:
                new_column_type = detect_column_type(new_column_name)
                if new_column_type in ["random_int", "pyfloat"]:
                    col_min_max = st.columns(2)
                    with col_min_max[0]:
                        min_val = st.number_input(f"Min {new_column_name}", value=0, key=f"min_{new_column_name}")
                    with col_min_max[1]:
                        max_val = st.number_input(f"Max {new_column_name}", value=100, key=f"max_{new_column_name}")
                    st.session_state.constraints[new_column_name] = (min_val, max_val)

            if st.button("Añadir Columna", key="añadir_columna", help="Añade la nueva columna al dataset"):
                try:
                    st.session_state.dataset[new_column_name] = generate_synthetic_data(
                        num_records=len(st.session_state.dataset), 
                        columns=[new_column_name],
                        constraints=st.session_state.constraints
                    )[new_column_name]

                    st.success(f"✅ Columna '{new_column_name}' añadida.")
                    st.dataframe(st.session_state.dataset, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error al añadir la columna: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # Función para convertir el DataFrame a CSV
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    # Opciones de descarga
    st.markdown("---")
    st.header("Descargar Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Botón para descargar CSV
        csv = convert_df_to_csv(st.session_state.dataset)
        st.download_button(
            label="📥 Descargar como CSV",
            data=csv,
            file_name="midas_synthetic_data.csv",
            mime="text/csv",
            help="Descarga el dataset generado en formato CSV"
        )
    
    with col2:
        # Botón para descargar Excel (opcional)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.dataset.to_excel(writer, sheet_name='Dataset', index=False)
        
        excel_data = buffer.getvalue()
        
        st.download_button(
            label="📥 Descargar como Excel",
            data=excel_data,
            file_name="midas_synthetic_data.xlsx",
            mime="application/vnd.ms-excel",
            help="Descarga el dataset generado en formato Excel"
        )