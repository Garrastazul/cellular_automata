# Autómatas Celulares 1D + Algoritmos Genéticos

Optimización de reglas de autómatas celulares unidimensionales mediante algoritmos genéticos.

## Descripción

Este proyecto implementa un sistema completo para:
- **Simular autómatas celulares 1D** con reglas binarias
- **Evolucionar reglas** mediante algoritmos genéticos
- **Evaluar convergencia** de poblaciones
- **Analizar entropía** de patrones emergentes

## Estructura del Proyecto

```
├── ac.py                      # Simulador de autómata celular (vectorizado)
├── genetic_algorithm.py       # Algoritmo genético completo (1000 generaciones)
├── utils.py                   # Funciones consolidadas (callR, randvec01, etc)
├── plot_automaton.py          # Visualización de patrones
├── evaluate_automaton.py      # Evaluación con 10,000 casos
├── evaluate_rule.py           # Validación de regla específica
├── program_rules.py           # Interfaz interactiva para reglas
├── Entropy/shannon_entropy.py # Cálculo de entropía
├── Proyecto.ipynb             # Notebook principal con ejemplos
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## Instalación

```bash
pip install -r requirements.txt
```

**Dependencias:**
- `numpy>=1.21.0` - Operaciones numéricas vectorizadas
- `matplotlib>=3.5.0` - Visualización de patrones

## Uso Rápido

### 1. Ejecutar Algoritmo Genético Completo
```bash
python genetic_algorithm.py
```
Genera 1000 generaciones de evolución y muestra el mejor individuo encontrado.

### 2. Usar en un Script Python
```python
from ac import ac
from utils import callR, randvec01
import numpy as np

# Crear una regla (Regla 30)
regla = callR(30, r=1)

# Condición inicial
I1 = np.zeros(101)
I1[50] = 1

# Simular 100 pasos
N = ac(regla, r=1, I1=I1, t=100)
```

### 3. Usar en Jupyter Notebook
```bash
jupyter notebook Proyecto.ipynb
```
Abre ejemplos ejecutables con explicaciones de código interno.

## Parámetros de Configuración

### Búsqueda Rápida (test)
```python
poblacion = 50
gen_max = 100
num_CI = 20
```
Tiempo: ~1 minuto

### Búsqueda Balanceada
```python
poblacion = 100
gen_max = 500
num_CI = 100
```
Tiempo: ~30 minutos

### Búsqueda Exhaustiva
```python
poblacion = 200
gen_max = 1000
num_CI = 200
```
Tiempo: ~2 horas

## Autómata Celular

**Función:** `ac(R, r, I1, t)`

- `R`: Regla (array binario de tamaño 2^(2r+1))
- `r`: Radio de vecindad (generalmente 1 o 3)
- `I1`: Condición inicial (array binario)
- `t`: Pasos de tiempo a simular

**Características:**
- Implementación vectorizada con NumPy
- Condiciones de frontera periódicas
- Precálculo de exponentes para eficiencia

## Algoritmo Genético

**Ciclo Generacional:**

1. **Evaluación** - Calcula nota de cada regla
2. **Selección** - Mantiene los 10 mejores (elitismo)
3. **Cruzamiento** - Combina genes de padres
4. **Mutación** - Invierte bits aleatoriamente (5%)
5. **Nueva generación** - Repite loop

**Objetivo:** Encontrar reglas que hagan convergir el autómata a estados específicos según la densidad inicial.

## Funciones Principales

### utils.py

| Función | Descripción |
|---------|-------------|
| `callR(nom, r)` | Convierte número decimal a representación binaria de regla |
| `randvec01(filas, columnas, P0)` | Genera matriz aleatoria binaria |
| `calpercent(A)` | Calcula porcentaje de 1s en cada fila |
| `numcoinc(A, B)` | Cuenta coincidencias entre dos arrays |
| `matcolon(A, B)` | Crea rangos para indexación |

### Entropy

| Función | Descripción |
|---------|-------------|
| `shannon_entropy(mat, d)` | Calcula entropía de Shannon en ventanas d×d |

## Resultados Esperados

- **Mejor nota**: 0.80-0.95 (80-95% de convergencia)
- **Generación de convergencia**: 200-500
- **Reglas encontradas**: Generalmente clasificadores simples o patrones emergentes

## Scripts Adicionales

| Script | Propósito |
|--------|-----------|
| `plot_automaton.py` | Visualiza 40 autómatas con diferentes reglas |
| `evaluate_automaton.py` | Evalúa regla con 10,000 condiciones iniciales (width=301) |
| `evaluate_rule.py` | Calcula nota de una regla específica |
| `program_rules.py` | Interfaz interactiva para programar reglas manualmente |

## Importabilidad

Todos los scripts tienen `if __name__ == "__main__":` para permitir importación sin ejecutar código:

```python
# Seguro para importar - no ejecuta lógica principal
from genetic_algorithm import CondicionesFinales
from ac import ac
from utils import *
```

## Notebook Jupyter

`Proyecto.ipynb` incluye:
- ✓ Ejemplos de todas las funciones
- ✓ Código interno de funciones principales
- ✓ Gráficos de convergencia
- ✓ Análisis de efectos de parámetros
- ✓ Comparativos de regulación entrópica
- ✓ Recomendaciones de configuración

## Requisitos

- Python 3.7+
- NumPy 1.21+
- Matplotlib 3.5+

## Mejoras Futuras

- [ ] Paralelización con multiprocessing
- [ ] Hill-climbing local adicional
- [ ] Validación cruzada de reglas encontradas
- [ ] Análisis de complejidad de patrones
- [ ] Visualización en tiempo real
- [ ] Exportar reglas encontradas

## Autor

Proyecto universitario - Estudio de autómatas celulares y evolución computacional

## Licencia

Uso académico permitido
