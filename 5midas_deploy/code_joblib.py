import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Datos de ejemplo con múltiples features
X_train = np.array([
    [25, 1.75, 70],   # Edad, altura (m), peso (kg)
    [30, 1.80, 80],
    [22, 1.65, 60],
    [40, 1.90, 90],
    [35, 1.78, 85],
    [28, 1.72, 75]
])
y_train = np.array([0, 1, 0, 1, 1, 0])  # 0 = No, 1 = Sí (ejemplo de clasificación)

# Crear un pipeline con escalado + modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression())
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(pipeline, "MIDAS/5midas_deploy/model.joblib")

print("Modelo guardado como 'multi_input_model.joblib'")
