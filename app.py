import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import io
from thefuzz import process  # Fuzzy matching para reconocer columnas similares
from sklearn.preprocessing import StandardScaler

# Definir la red neuronal
class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=0.3)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, inputs):
        x = self.relu(self.bn1(self.linear1(inputs)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.dropout2(x)
        output = self.linear3(x)
        return output

# Función para convertir valores de texto en números
def convertir_objetos_a_numerico(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace(',', '.', regex=True)  # Reemplazar comas por puntos
        df[col] = df[col].str.strip()  # Eliminar espacios en blanco
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir a numérico
    return df

# Función para corregir nombres de columnas usando fuzzy matching
def corregir_nombres_columnas(columnas_usuario, columnas_correctas):
    columnas_corregidas = {}
    for col in columnas_usuario:
        match, score = process.extractOne(col, columnas_correctas)  # Encuentra la mejor coincidencia
        if score > 80:  # Umbral de similitud (ajustable)
            columnas_corregidas[col] = match
    return columnas_corregidas

# Cargar dataset de referencia para obtener estadísticas de escalado
file_path = "produccion_limpia.csv"
data = pd.read_csv(file_path, sep=";")
data = convertir_objetos_a_numerico(data)

columnas_entrada = [col for col in data.columns if col != "Prod. Total"]
n_entradas = len(columnas_entrada)

# Escalado de datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = data[columnas_entrada].values
y_train = data["Prod. Total"].values.reshape(-1, 1)

scaler_X.fit(X_train)
scaler_y.fit(y_train)

# Cargar modelo entrenado
modelo = Red(n_entradas)
modelo.load_state_dict(torch.load("modelo_entrenado.pth"))
modelo.eval()

# Interfaz en Streamlit
st.title("Predicción de Producción Total")
with st.expander("¿Como funciona?", expanded=True):
    st.markdown("""
    # Descripción de la Aplicación
    
    Esta aplicación implementa una red neuronal en **PyTorch** para resolver un problema de regresión. Su objetivo es predecir una variable numérica a partir de un conjunto de datos estructurados. La arquitectura de la red está diseñada para mejorar la precisión y la estabilidad del entrenamiento mediante varias técnicas avanzadas.
    
    ## 🛠️ Características principales:
    - **🔗 Red Neuronal Profunda**: Arquitectura de tres capas completamente conectadas con 128 y 64 neuronas ocultas.
    - **🛡️ Regularización**: Incluye **Batch Normalization** y **Dropout** para evitar el sobreajuste.
    - **⚡ Optimización Avanzada**: Uso del optimizador **Adam** con regularización L2 y ajuste de la tasa de aprendizaje con **ReduceLROnPlateau**.
    - **📏 Escalado de Datos**: Normalización de variables predictoras y de la variable objetivo para mejorar la estabilidad del entrenamiento.
    - **📊 Evaluación Continua**: Cálculo de métricas como **MSE (Error Cuadrático Medio)** y **R² (Coeficiente de Determinación)** en el conjunto de prueba.
    - **💾 Almacenamiento de Resultados**: Guardado del historial de entrenamiento y las predicciones desescaladas para análisis posterior.
    
    ## 🔄 Flujo de la Aplicación:
    1. **Preprocesamiento de Datos**: Normalización de variables y división en conjuntos de entrenamiento y prueba.
    2. **Entrenamiento del Modelo**: Uso de descenso de gradiente con retropropagación.
    3. **Evaluación y Ajuste**: Medición del rendimiento en el conjunto de prueba y ajuste dinámico de la tasa de aprendizaje.
    4. **Predicciones Finales**: Desescalado de predicciones y almacenamiento en un archivo CSV para su interpretación.
    
    ---
    📌 *Desarrollado con PyTorch y Streamlit*
    """)
st.write("Sube un archivo Excel con los datos de entrada para obtener las predicciones.")

# Cargar archivo Excel
archivo = st.file_uploader("Sube un archivo Excel", type=["xls", "xlsx"])

if archivo:
    df = pd.read_excel(archivo)
    df = convertir_objetos_a_numerico(df)

    # Validar y corregir nombres de columnas
    columnas_corregidas = corregir_nombres_columnas(df.columns, columnas_entrada)
    
    # Aplicar los nombres corregidos
    df.rename(columns=columnas_corregidas, inplace=True)
    
    # Revisar si todas las columnas necesarias están presentes
    if set(columnas_entrada).issubset(df.columns):
        df = df[columnas_entrada]  # Seleccionar solo las columnas de entrada
        
        # Escalar los datos de entrada
        X_scaled = scaler_X.transform(df.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Realizar predicciones escaladas
        with torch.no_grad():
            y_scaled_pred = modelo(X_tensor).numpy().flatten()

        # Desescalar las predicciones
        y_pred_desescalado = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()

        # Agregar predicciones al DataFrame
        df["Producción Total Estimada"] = y_pred_desescalado

        # Guardar el DataFrame con predicciones en un archivo Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Predicciones")
        output.seek(0)

        # Botón para descargar el archivo Excel
        st.download_button(
            label="Descargar Excel con predicciones",
            data=output,
            file_name="predicciones.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("Predicciones generadas con éxito. Descarga el archivo con el botón de arriba.")
    else:
        st.error(f"Faltan columnas requeridas: {set(columnas_entrada) - set(df.columns)}. Verifica el archivo.")

