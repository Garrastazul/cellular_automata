# Evolución de Autómatas Celulares mediante Algoritmos Genéticos

Este proyecto implementa un simulador de Autómatas Celulares (AC) unidimensionales y un marco de trabajo de Algoritmos Genéticos (AG) diseñado para evolucionar reglas que resuelvan tareas específicas de procesamiento de información.

## 1. CellularAutomaton

La clase `CellularAutomaton` es el motor de simulación. Representa un autómata celular 1D donde el estado de una célula en el tiempo $t+1$ depende de su vecindad en el tiempo $t$.

### Mecánica de Simulación
* **Vecindad ($r$):** El radio define cuántas células a la izquierda y derecha se consideran. Una vecindad completa tiene un tamaño de $2r + 1$.
* **Espacio de Reglas:** Para un radio $r$, existen $2^{2r+1}$ configuraciones de vecindad posibles. La regla se almacena como un vector binario que mapea cada configuración a un estado de salida (0 o 1).
* **Condiciones de Frontera:** Se implementan condiciones periódicas (toroidales) utilizando desplazamientos circulares (`np.roll`), lo que permite que la información que sale por un extremo reingrese por el opuesto.

### Métodos Principales
* `run(initial_state)`: Ejecuta la simulación paso a paso y almacena la trayectoria completa en memoria.
* `_run(initial_state)`: Versión optimizada que solo devuelve el estado final, utilizada intensivamente durante la evaluación del algoritmo genético.
* `visualize()`: Genera un diagrama de espacio-tiempo utilizando Matplotlib para observar la evolución de los patrones.



## 2. GeneticAlgorithm

La clase `GeneticAlgorithm` actúa como el motor de optimización. Su objetivo es encontrar una regla (cromosoma) que transforme un conjunto de condiciones iniciales en estados finales deseados.

### Ciclo Evolutivo
1. **Evaluación (Fitness):** Se calcula el desempeño de cada individuo comparando el estado final alcanzado tras $N$ pasos contra un objetivo predefinido. La nota es la media de aciertos bit a bit sobre múltiples condiciones iniciales.
2. **Selección:** Se utiliza una estrategia de elitismo donde los mejores individuos (`num_parents`) pasan directamente a la siguiente generación y actúan como base para la reproducción.
3. **Crossover:** Se seleccionan dos padres al azar y se combinan sus reglas mediante un punto de corte aleatorio (single-point crossover) para crear un descendiente.
4. **Mutación:** Cada bit de la regla del descendiente tiene una probabilidad `p_mutacion` de invertirse. Esto permite mantener la diversidad y explorar nuevas regiones del espacio de búsqueda.

### Métricas de Control
* **Fitness (Mejor/Promedio):** Indica el progreso hacia la solución óptima.
* **Diversidad Genética:** Calcula la distancia de Hamming promedio entre las reglas de la población para detectar convergencia prematura.

## 3. Estructura del Proyecto

* **Clases Principales:** Definidas en el script principal para la ejecución de la lógica AC y AG.
* **Jupyter Notebook:** Contiene experimentos interactivos, análisis detallados de reglas evolucionadas y visualizaciones de la dinámica del sistema.
* **Tests:** Suite de pruebas unitarias para validar la correcta aplicación de las reglas del AC y la integridad de los operadores genéticos (cruce y mutación).
* **utils.py:** Proporciona funciones auxiliares para la generación de vectores aleatorios y la definición de condiciones de contorno/objetivos.

## 4. Requisitos

El proyecto requiere las siguientes librerías de Python:
* `numpy`: Procesamiento de matrices y lógica de vecindad.
* `matplotlib`: Visualización de trayectorias y estadísticas de evolución.
* `numba`: Preparado para aceleración JIT de los bucles de simulación.