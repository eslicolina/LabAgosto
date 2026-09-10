# LabAgosto
# Gestión Ágil de Proyectos de IA Aplicada a la Detección de Sesgo en Evaluaciones Crediticias

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)
![Methodology](https://img.shields.io/badge/Methodology-Agile%20%2F%20MLOps-green.svg)
![Compliance](https://img.shields.io/badge/Compliance-EU%20AI%20Act%20(High--Risk)-red.svg)

## 📌 Descripción del Proyecto

Este repositorio contiene la implementación práctica y la auditoría de equidad algorítmica para un sistema predictivo de evaluación crediticia. El proyecto integra un **modelo de aprendizaje automático** (Regresión Logística) con marcos de **Gestión Ágil de Proyectos** (Scrum/Kanban) y **principios de MLOps**, orientados a identificar y auditar el **sesgo algorítmico** en escenarios financieros de alto riesgo.

El objetivo principal es evidenciar el dilema entre las métricas de rendimiento técnico tradicional ($Accuracy$) y las métricas de responsabilidad ética y legal ($Disparate\ Impact$), alineándose con el marco regulatorio de la **Ley de Inteligencia Artificial de la Unión Europea (2024)**.

---

## 🎯 Objetivos

1. **Simular y Preprocesar Datos Financieros:** Construcción de un dataset sintético ($N = 1,000$) con variables socioeconómicas y atributos protegidos.
2. **Entrenar Modelo Predictivo:** Clasificación binaria de elegibilidad crediticia utilizando Regresión Logística y estandarización de variables.
3. **Auditar la Equidad Algorítmica:** Aplicación de la **Regla de las cuatro quintas partes (Regla del 80%)** mediante el cálculo del *Impacto Dispar* ($DI$).
4. **Implementar Gestión Ágil (PMI Agile / MLOps):** Estructurar el desarrollo por ciclos (*Sprints*), control de flujo continuo (Kanban con límites WIP) y gestión proactiva de deuda técnica ética.

---

## 🔬 Resultados Empíricos

El experimento ejecutado arrojó los siguientes resultados cuantitativos:

| Dimensión de Evaluación | Métrica Evaluada | Resultado Obtenido | Umbral / Criterio | Diagnóstico |
| :--- | :--- | :--- | :--- | :--- |
| **Rendimiento Técnico** | Exactitud Global (*Accuracy*) | **99.00%** | $\ge 80.00\%$ | **Aprobado** (Alta precisión) |
| **Rendimiento Técnico** | $F_1$-Score (Clase 0 / Clase 1) | **1.00 / 0.99** | $\ge 0.80$ | **Aprobado** (Consistente) |
| **Evaluación Ética** | Impacto Dispar ($DI$) | **0.1481** | $\ge 0.8000$ | **RECHAZADO (Sesgo Severo)** |

### ⚠️ Hallazgo Clave
* **Tasa de Aprobación Grupo No Protegido:** $67.50\%$
* **Tasa de Aprobación Grupo Protegido:** $10.00\%$
* **Índice $DI$:** $0.1481$ ($14.81\% < 80.00\%$)

> **Conclusión:** Un modelo con $99\%$ de precisión técnica puede ser altamente discriminatorio e inaceptable para producción bajo regulaciones de IA de Alto Riesgo. La auditoría ética en ciclos ágiles permite bloquear el despliegue a tiempo.

---

## 🛠️ Arquitectura de Gestión Ágil

- **Sprint 1 (Scrum):** Ingesta, limpieza de datos y entrenamiento del modelo base.
- **Sprint 2 (Scrum):** Auditoría ética y cálculo de impacto dispar.
- **Kanban & WIP Limits:** Restricción de tareas en paralelo en la fase de validación para prevenir congestión.
- **Deuda Técnica Ética:** Asignación del $20\%$ de capacidad por ciclo para refactorización e investigación de sesgo.

---

## 💻 Instalación y Ejecución

### Prerrequisitos
- Python 3.10 o superior
- Bibliotecas: `numpy`, `pandas`, `scikit-learn`

### Pasos de Ejecución

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/eslicolina/LabAgosto.git](https://github.com/eslicolina/LabAgosto.git)
   cd LabAgosto
Instalar dependencias:

Bash
pip install numpy pandas scikit-learn
Ejecutar el script principal:

Bash
python Metodo_agil.py