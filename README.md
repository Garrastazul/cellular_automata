# Autómatas Celulares 1D + Algoritmo Genético (POO)

Versión consolidada con **2 archivos únicamente** (sin dependencias innecesarias):

- **`models.py`** - Clases `CellularAutomaton` y `GeneticAlgorithm` (~500 líneas)
- **`utils.py`** - Funciones auxiliares (~150 líneas)

## 🚀 Uso Rápido

### Opción 1: Ejecutar AG Directamente
```bash
python models.py
```
✅ Ejecuta 50 generaciones automáticamente  
✅ Muestra evolución en terminal  
✅ Genera 4 gráficos automáticos  
✅ Imprime análisis del mejor CA  

### Opción 2: Usar en tu Python
```python
from models import GeneticAlgorithm

ag = GeneticAlgorithm(poblacion=50, gen_max=50)
ag.evolucionar()
ag.visualizar_resultados()
ag.analizar_mejor_individuo()
```

### Opción 3: Personalizar Parámetros
Edita el bloque `if __name__ == "__main__":` en `models.py`:
```python
ag = GeneticAlgorithm(
    poblacion=100,          # Individuos por generación
    num_parents=10,         # Élite
    gen_max=100,            # Generaciones totales
    num_CI=50,              # Condiciones iniciales
    ancho=101,              # Ancho del autómata
    r=3,                    # Radio (3 = 128 bits)
    p_mutacion=0.05         # Tasa de mutación
)
```

## 📁 Estructura Consolidada

```
Proyecto POO (Mínimo)
├── models.py            ⭐⭐⭐ PRINCIPAL
│   ├── Clase CellularAutomaton
│   │   ├── _run()           ← Simulación AC
│   │   ├── _mutate()        ← Mutación genética
│   │   ├── evaluate()       ← Calcular fitness
│   │   ├── crossover()      ← Cruzamiento
│   │   └── copy()           ← Copiar CA
│   │
│   ├── Clase GeneticAlgorithm
│   │   ├── evolucionar()              ← Ejecutar AG
│   │   ├── visualizar_resultados()    ← 4 gráficos
│   │   └── analizar_mejor_individuo() ← Análisis
│   │
│   └── if __name__ == "__main__":   ← Punto de entrada
│
└── utils.py             ⭐ AUXILIARES
    ├── randvec01()              ← Matriz binaria aleatoria
    ├── CondicionesFinales()     ← Generar objetivo
    ├── callR()                  ← Decimal → binario
    ├── calpercent()             ← Densidad por fila
    ├── numcoinc()               ← Coincidencias
    ├── matcolon()               ← Generador rangos
    └── shannon_entropy()        ← Entropía de Shannon
```

## 🎯 Clases Principales

### CellularAutomaton
Simula autómatas celulares 1D con reglas binarias.

```python
from models import CellularAutomaton
import numpy as np

# Crear CA con regla aleatoria
ca = CellularAutomaton(
    rule=np.random.randint(0, 2, 128),
    r=3,
    width=101,
    timesteps=202
)

# Ejecutar simulación
ic = np.random.randint(0, 2, 101)
evolucion = ca._run(ic)  # Shape: (202, 101)

# Operadores genéticos
regla_mutada = ca._mutate(0.05)
ca2 = CellularAutomaton(np.random.randint(0, 2, 128), r=3, width=101)
ca_hijo = ca.crossover(ca2)

# Evaluar fitness
A = np.random.randint(0, 2, (5, 101))
Cf = np.random.randint(0, 2, (5, 101))
fitness = ca.evaluate(A, Cf)
```

### GeneticAlgorithm
Evoluciona poblaciones de autómatas.

```python
from models import GeneticAlgorithm

# Crear AG
ag = GeneticAlgorithm(
    poblacion=50,
    num_parents=5,
    gen_max=50,
    num_CI=20
)

# Ejecutar
tiempo = ag.evolucionar(verbose=True)

# Visualizar
ag.visualizar_resultados()

# Analizar
ag.analizar_mejor_individuo()

# Acceder estadísticas
print(f"Mejor: {max(ag.estadisticas['mejor']):.4f}")
print(f"Diversidad final: {ag.estadisticas['diversidad'][-1]:.4f}")
```

## 📦 Instalación

```bash
# Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# Instalar dependencias
pip install numpy matplotlib
```

## 📊 Salida Esperada

```
╔════════════════════════════════════════════════════════════╗
║          ALGORITMO GENÉTICO - AUTÓMATAS CELULARES         ║
╚════════════════════════════════════════════════════════════╝

Parámetros:
  Población: 50 | Parents: 5 | Generaciones: 50
  Condiciones iniciales: 20 | Ancho: 101 | Radio: 3

Evolucionando...
Gen   1/50 | Mejor: 0.5200 | Prom: 0.3456 | Div: 0.412
Gen  10/50 | Mejor: 0.7450 | Prom: 0.6234 | Div: 0.295
Gen  20/50 | Mejor: 0.8650 | Prom: 0.7456 | Div: 0.182
Gen  30/50 | Mejor: 0.9100 | Prom: 0.8234 | Div: 0.095
Gen  50/50 | Mejor: 0.9500 | Prom: 0.8789 | Div: 0.064

✓ EVOLUCIÓN COMPLETADA
Tiempo: 32.45 segundos

MEJOR AUTÓMATA
Regla: 110100011010000111...
Convergencia: 95% ✓

[4 gráficos generados automáticamente]
```

## 🔧 Parámetros

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `poblacion` | 50 | ↑ Mejor calidad, ↑ tiempo |
| `num_parents` | 5 | Células que persisten (élite) |
| `gen_max` | 50 | ↑ Mayor convergencia |
| `num_CI` | 20 | Condiciones iniciales para evaluar |
| `p_mutacion` | 0.05 | ↑ Mayor exploración |

## ⚡ Rendimiento

| Config | Población | Generaciones | Tiempo |
|--------|-----------|-------------|--------|
| Rápida | 30 | 20 | ~5 seg |
| Normal | 50 | 50 | ~30 seg |
| Completa | 100 | 100 | ~5 min |

## 💻 Requisitos

- Python 3.9+
- NumPy 1.24+
- Matplotlib 3.7+

## 🎯 Ejemplos

### 1. Búsqueda Rápida
```bash
python models.py
```

### 2. Análisis Personalizado
```python
from models import GeneticAlgorithm
import numpy as np

np.random.seed(42)
ag = GeneticAlgorithm(poblacion=100, gen_max=100)
ag.evolucionar(verbose=True)
ag.visualizar_resultados()
```

### 3. Múltiples Ejecuciones
```python
from models import GeneticAlgorithm
import numpy as np

resultados = []
for seed in range(10):
    np.random.seed(seed)
    ag = GeneticAlgorithm()
    ag.evolucionar()
    resultados.append(max(ag.estadisticas['mejor']))

print(f"Promedio: {np.mean(resultados):.4f}")
print(f"Mejor: {np.max(resultados):.4f}")
```

## 🔍 API Rápida

### CellularAutomaton
```python
ca._run(initial_state)              # → evolucion (matriz)
ca._mutate(0.05)                    # → regla_mutada
ca.evaluate(A, Cf)                  # → fitness (0-1)
ca.crossover(other)                 # → hijo (CA)
ca.copy()                            # → copia profunda
```

### GeneticAlgorithm
```python
ag.evolucionar(verbose=True)        # → tiempo_total
ag.visualizar_resultados()          # → muestra 4 gráficos
ag.analizar_mejor_individuo()       # → imprime análisis
ag.estadisticas['mejor']            # → array de mejores
ag.mejores_individuos[-1]           # → mejor CA final
```

### Utils
```python
randvec01(filas, cols, P0)          # → matriz binaria
CondicionesFinales(num_CI, ancho)   # → (A, Cf) tupla
callR(decimal, r)                   # → regla binaria
calpercent(A)                        # → densidades
shannon_entropy(mat, d)              # → H (bits)
```

## 🐛 Troubleshooting

| Problema | Solución |
|----------|----------|
| Bajo fitness | ↑ `poblacion`, ↑ `gen_max` |
| Convergencia lenta | ↑ `p_mutacion` |
| Lentitud extrema | ↓ `num_CI` |
| Resultados inconsistentes | Usar `np.random.seed()` |

## 📄 Licencia

Código educativo para investigación en Autómatas Celulares.

---

**Versión:** 3.0 (Consolidada)  
**Estado:** ✅ Completo  
**Archivos:** 2 (models.py, utils.py)  
**Total:** ~650 líneas
