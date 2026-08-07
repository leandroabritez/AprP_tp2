# TP2 — Aprendizaje Profundo · CEIA · Cohorte 2025

**Alumno:** Leandro Britez  
**Materia:** Aprendizaje Profundo  
**Institución:** CEIA — Centro de Especialización en Inteligencia Artificial  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leandroabritez/AprP_tp2/blob/main/BRITEZ-LEANDRO-TP2-Co25.ipynb)

---

## Objetivo del trabajo

El objetivo de este trabajo práctico es construir y comparar modelos de clasificación binaria sobre el dataset **Adult Census Income** para predecir si el ingreso anual de una persona supera los USD 50.000. El eje central es el uso de **redes neuronales con capas de embedding** para el tratamiento de variables categóricas de alta cardinalidad, implementadas en **PyTorch**.

---

## Dataset

**Adult Census Income** — extraído del repositorio UCI Machine Learning.

| Característica | Detalle |
|---|---|
| Registros | ~48.842 (train + val) |
| Target | `income`: `<=50K` / `>50K` |
| Desbalanceo | ~75% clase negativa / ~25% clase positiva |

### Variables

| Tipo | Features |
|---|---|
| **Numéricas** | `age`, `fnlwgt`, `educational-num`, `capital-gain`, `capital-loss`, `hours-per-week` |
| **Categóricas OHE** | `workclass`, `marital-status`, `relationship`, `race`, `sex` |
| **Ordinal** | `education` (orden educativo de Preschool a Doctorate) |
| **Embeddings** | `occupation`, `skill-profile`, `native-country` |

> `skill-profile` es una feature de ingeniería que agrupa ocupaciones en perfiles de habilidad.

---

## Estructura del notebook

```
BRITEZ-LEANDRO-TP2-Co25.ipynb
│
├── 0. Setup e imports
│
├── a) Análisis y preprocesamiento (3 pts)
│   ├── EDA: distribuciones, correlaciones, valores nulos y outliers
│   ├── Ingeniería de features (skill-profile, has_capital_*)
│   ├── Justificación de transformaciones por tipo de variable
│   └── Pipeline: StandardScaler + OHE + OrdinalEncoder + LabelEncoders
│
└── b) Modelo con Embeddings (3 pts)
    ├── Arquitectura: Embedding(occupation) + Embedding(skill-profile)
    │               + Embedding(native-country) + MLP con Dropout
    ├── Optimizer: Adam + CosineAnnealingLR
    ├── Loss: BCEWithLogitsLoss (clasificación binaria)
    ├── Curvas: Accuracy y F1-Macro vs Epoch (train / validación)
    └── Evaluación: Classification Report + Matrices de Confusión
```

---

## Decisiones de diseño de embeddings

| Feature | Cardinalidad | Dim embedding | Justificación |
|---|---|---|---|
| `occupation` | ~15 | 7–8 | Grupos laborales con semántica, dim moderada |
| `skill-profile` | ~10 | 6 | Feature sintética, cardinalidad reducida |
| `native-country` | ~41 | 21 | Alta cardinalidad + relaciones geográficas/culturales |

**Regla aplicada:** `dim = max(4, min(50, ceil(n_categorías / 2)))`

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellowgreen?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.x-blue?logo=pandas)

---

## Cómo ejecutar

### En Google Colab
Hacé clic en el badge **Open in Colab** al inicio de este README.

### En local (Python 3.11)
```bash
# Clonar el repositorio
git clone https://github.com/leandroabritez/AprP_tp2.git
cd AprP_tp2

# Instalar dependencias
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn matplotlib seaborn

# Abrir el notebook
jupyter notebook BRITEZ-LEANDRO-TP2-Co25.ipynb
```