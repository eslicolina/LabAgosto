import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# ==========================================
# FASE 1: SIMULACIÓN DEL CONJUNTO DE DATOS
# ==========================================
np.random.seed(42)
n_muestras = 1000

# Atributo protegido: 0 = Grupo No Protegido, 1 = Grupo Protegido
grupo_protegido = np.random.binomial(1, 0.4, n_muestras)
ingresos = np.random.normal(50000, 15000, n_muestras)
historial_crediticio = np.random.uniform(300, 850, n_muestras)

# Ajuste de coeficientes para balancear clases (0 y 1)
z = (0.00003 * ingresos + 0.004 * historial_crediticio - 1.2 * grupo_protegido - 3.5)
prob_aprobacion = 1 / (1 + np.exp(-z))
elegible = (prob_aprobacion > 0.5).astype(int)

# Crear DataFrame
df = pd.DataFrame({
    'Grupo_Protegido': grupo_protegido,
    'Ingresos': ingresos,
    'Historial_Crediticio': historial_crediticio,
    'Elegible': elegible
})

# ==========================================
# FASE 2: PREPARACIÓN Y DIVISIÓN DE DATOS
# ==========================================
X = df[['Grupo_Protegido', 'Ingresos', 'Historial_Crediticio']]
y = df['Elegible']

# División 80% entrenamiento / 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalamiento de variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# FASE 3 Y 4: MODELADO Y PREDICCIÓN
# ==========================================
modelo = LogisticRegression(random_state=42)
modelo.fit(X_train_scaled, y_train)

# --- AQUÍ ESTABA LA LÍNEA FALTANTE ---
y_pred = modelo.predict(X_test_scaled)
# --------------------------------------

# ==========================================
# FASE 4: EVALUACIÓN DE MÉTRICAS Y AUDITORÍA
# ==========================================
print("==========================================")
print("1. INFORME DE RENDIMIENTO TÉCNICO DEL MODELO")
print("==========================================")
print(classification_report(y_test, y_pred))

# Auditoría Ética: Evaluación de Impacto Dispar (DI)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df['Prediccion'] = y_pred

# Tasa de aprobación por grupo
tasa_protegido = X_test_df[X_test_df['Grupo_Protegido'] == 1]['Prediccion'].mean()
tasa_no_protegido = X_test_df[X_test_df['Grupo_Protegido'] == 0]['Prediccion'].mean()

# Cálculo del Impacto Dispar (DI)
impacto_dispar = tasa_protegido / tasa_no_protegido if tasa_no_protegido > 0 else 0

print("==========================================")
print("2. AUDITORÍA ÉTICA DE SESGO ALGORÍTMICO")
print("==========================================")
print(f"Tasa de Aprobación - Grupo Protegido:    {tasa_protegido:.2%}")
print(f"Tasa de Aprobación - Grupo No Protegido: {tasa_no_protegido:.2%}")
print(f"Índice de Impacto Dispar (DI):          {impacto_dispar:.4f}")
print("------------------------------------------")

if impacto_dispar < 0.80:
    print("DIAGNÓSTICO: ALERTA DE SESGO DETECTADA.")
    print("El resultado está por debajo del umbral de 0.80 (Regla de las cuatro quintas partes).")
else:
    print("DIAGNÓSTICO: MODELO EQUITATIVO.")
    print("El modelo cumple con el criterio de equidad (DI >= 0.80).")