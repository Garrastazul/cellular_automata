"""
Modelos de Autómata Celular y Algoritmo Genético
Programación Orientada a Objetos (POO)

Clases principales:
  • CellularAutomaton - Simulador de autómata celular 1D
  • GeneticAlgorithm - Algoritmo genético para evolucionar reglas
"""
import numpy as np
import matplotlib.pyplot as plt
import time


class CellularAutomaton:
    """Clase para autómata celular 1D con evolución genética"""
    
    def __init__(self, rule, r=3, width=101, timesteps=None):
        """
        Inicializa el autómata celular
        
        Parameters:
            rule: array de bits (regla binaria del AC)
            r: radio de vecindad (default: 3)
            width: ancho del autómata (default: 101)
            timesteps: número de pasos de evolución (default: 2*width)
        """
        self.rule = rule.astype(int)
        self.r = r
        self.width = width
        self.timesteps = timesteps if timesteps else 2 * width
    
    def _run(self, initial_state):
        """
        MÉTODO PRINCIPAL: Ejecuta la simulación del autómata
        
        Evoluciona la condición inicial durante 'timesteps' generaciones
        usando vecindades de radio r.
        
        Parameters:
            initial_state: array binario de condición inicial
            
        Returns:
            evolution: matriz (timesteps × width) con evolución temporal
        """
        AN = len(initial_state)
        M = np.zeros((self.timesteps, AN + 2 * self.r))
        
        # Boundary conditions: condiciones periódicas
        M[0, :] = np.concatenate([initial_state[(AN - self.r):], 
                                   initial_state, 
                                   initial_state[:self.r]])
        
        # Exponentes para cálculo de índice
        exp = 2.0 ** np.arange(2 * self.r, -1, -1)
        
        # Evolución temporal
        for i in range(1, self.timesteps):
            for j in range(self.r, AN + self.r):
                vecindad = M[i-1, (j - self.r):(j + self.r + 1)]
                indice = int(np.sum(vecindad * exp))
                M[i, j] = self.rule[len(self.rule) - 1 - indice]
            
            # Aplicar condiciones de frontera periódicas
            M[i, :self.r] = M[i, AN:(AN + self.r)]
            M[i, (AN + self.r):] = M[i, self.r:(2 * self.r)]
        
        # Extraer región central
        evolution = M[:, self.r:(AN + self.r)]
        return evolution
    
    def _mutate(self, mutation_rate=0.05):
        """
        MÉTODO PRINCIPAL: Aplica mutación genética a la regla
        
        Invierte aleatoriamente bits con probabilidad mutation_rate.
        
        Parameters:
            mutation_rate: probabilidad de mutación por bit (default: 0.05)
            
        Returns:
            regla_mutada: nueva regla (no modifica el original)
        """
        mutations = np.random.rand(len(self.rule)) < mutation_rate
        mutated_rule = np.mod(self.rule + mutations, 2)
        return mutated_rule
    
    def evaluate(self, initial_conditions, objectives):
        """
        Evalúa el desempeño de la regla
        
        Simula la regla usando múltiples condiciones iniciales
        y cuenta coincidencias con objetivos.
        
        Parameters:
            initial_conditions: matriz (num_CI × width) de condiciones iniciales
            objectives: matriz (num_CI × width) de estados objetivo
            
        Returns:
            fitness: puntuación normalizada [0, 1]
        """
        num_CI = len(initial_conditions)
        coincidencias = 0
        
        for i in range(num_CI):
            evolution = self._run(initial_conditions[i, :])
            final_state = evolution[-1, :]
            coincidencias += self._numcoinc(final_state, objectives[i, :])
        
        fitness = coincidencias / (num_CI * self.width)
        return fitness
    
    def crossover(self, other, crossover_point=None):
        """
        Recombinación genética: two-point crossover
        
        Combina reglas de dos autómatas en un punto aleatorio.
        
        Parameters:
            other: otro CellularAutomaton (padre 2)
            crossover_point: punto de cruce (aleatorio si None)
            
        Returns:
            hijo: nuevo CellularAutomaton result del cruzamiento
        """
        if crossover_point is None:
            crossover_point = np.random.randint(0, len(self.rule))
        
        child_rule = np.concatenate([self.rule[:crossover_point],
                                     other.rule[crossover_point:]])
        return CellularAutomaton(child_rule, self.r, self.width, self.timesteps)
    
    def copy(self):
        """Crea una copia profunda del autómata"""
        return CellularAutomaton(self.rule.copy(), self.r, self.width, self.timesteps)
    
    @staticmethod
    def _numcoinc(A, B):
        """Cuenta coincidencias elemento a elemento entre arrays"""
        A = np.array(A).flatten()
        B = np.array(B).flatten()
        return np.sum(A == B)
    
    def __repr__(self):
        rule_str = ''.join(self.rule.astype(int).astype(str))[:8] + '...'
        return f"CA(r={self.r}, rule_bits={len(self.rule)}, {rule_str})"
    
    def __str__(self):
        return f"CellularAutomaton: r={self.r}, width={self.width}, timesteps={self.timesteps}"


class GeneticAlgorithm:
    """Algoritmo genético para evolucionar autómatas celulares"""
    
    def __init__(self, poblacion=50, num_parents=5, gen_max=50,
                 num_CI=20, ancho=101, r=3, p_mutacion=0.05):
        """
        Inicializa el algoritmo genético
        
        Parameters:
            poblacion: tamaño de población
            num_parents: número de padres que pasan a siguiente generación
            gen_max: número máximo de generaciones
            num_CI: número de condiciones iniciales para evaluación
            ancho: ancho del autómata
            r: radio de vecindad
            p_mutacion: probabilidad de mutación por bit
        """
        from utils import randvec01
        
        self.poblacion = poblacion
        self.num_parents = num_parents
        self.gen_max = gen_max
        self.num_CI = num_CI
        self.ancho = ancho
        self.r = r
        self.p_mutacion = p_mutacion
        self.lR = 2 ** (2 * r + 1)
        
        # Inicializar población
        self.poblacion_ca = [
            CellularAutomaton(randvec01(1, self.lR, 0.5)[0], r=r, width=ancho)
            for _ in range(poblacion)
        ]
        
        # Tracking de evolución
        self.mejor_nota_generaciones = np.zeros(gen_max)
        self.mejores_individuos = []
        self.estadisticas = {
            'mejor': [],
            'promedio': [],
            'peor': [],
            'diversidad': []
        }
    
    def _evaluar_poblacion(self, A, Cf):
        """Evalúa el fitness de todos los individuos"""
        notas = np.array([ca.evaluate(A, Cf) for ca in self.poblacion_ca])
        return notas
    
    def _calcular_diversidad(self):
        """Calcula la diversidad genética de la población"""
        if len(self.poblacion_ca) < 2:
            return 0.0
        
        diferencias = 0
        total_comparaciones = 0
        
        for i in range(min(len(self.poblacion_ca), 10)):
            for j in range(i + 1, min(len(self.poblacion_ca), 10)):
                diferencias += np.sum(self.poblacion_ca[i].rule != self.poblacion_ca[j].rule)
                total_comparaciones += self.lR
        
        if total_comparaciones == 0:
            return 0.0
        return diferencias / total_comparaciones
    
    def evolucionar(self, verbose=True):
        """Ejecuta el algoritmo genético"""
        from utils import CondicionesFinales, randvec01
        
        print("=" * 75)
        print(f"ALGORITMO GENÉTICO (POO) - {self.gen_max} GENERACIONES")
        print("=" * 75)
        print(f"Población: {self.poblacion} | Padres: {self.num_parents} | "
              f"Mutación: {self.p_mutacion * 100:.1f}%")
        print(f"CI: {self.num_CI} | Ancho: {self.ancho} | Radio: {self.r}")
        print(f"Tamaño regla: {self.lR} bits | Generaciones: {self.gen_max}\n")
        
        inicio = time.time()
        
        for g in range(self.gen_max):
            # 1. Generar condiciones iniciales y objetivos
            A, Cf = CondicionesFinales(self.num_CI, self.ancho)
            
            # 2. Evaluar población
            notas = self._evaluar_poblacion(A, Cf)
            
            # 3. Guardar estadísticas
            self.mejor_nota_generaciones[g] = np.max(notas)
            self.estadisticas['mejor'].append(np.max(notas))
            self.estadisticas['promedio'].append(np.mean(notas))
            self.estadisticas['peor'].append(np.min(notas))
            self.estadisticas['diversidad'].append(self._calcular_diversidad())
            
            # 4. Selección (Elitismo)
            indices_ordenados = np.argsort(-notas)
            mejores_idx = indices_ordenados[:self.num_parents]
            padres = [self.poblacion_ca[i].copy() for i in mejores_idx]
            
            # Guardar mejor individuo si mejoró
            if g == 0 or self.mejor_nota_generaciones[g] > self.mejor_nota_generaciones[g - 1]:
                self.mejores_individuos.append(padres[0].copy())
            
            # 5. Crear nueva población (cruzamiento y mutación)
            nueva_poblacion = padres.copy()  # Mantener padres (elitismo)
            
            for k in range(self.poblacion - self.num_parents):
                # Seleccionar dos padres aleatorios
                padre1 = padres[np.random.randint(0, self.num_parents)]
                padre2 = padres[np.random.randint(0, self.num_parents)]
                
                # Cruzamiento
                hijo = padre1.crossover(padre2)
                
                # Mutación
                hijo.rule = hijo._mutate(self.p_mutacion)
                
                nueva_poblacion.append(hijo)
            
            self.poblacion_ca = nueva_poblacion
            
            # Imprimir progreso
            if verbose and ((g + 1) % 10 == 0 or g == 0):
                div_str = f"Div: {self.estadisticas['diversidad'][-1]:.3f}" if g > 0 else "Inicializando"
                print(f"Gen {g+1:3d}/{self.gen_max} | Mejor: {self.mejor_nota_generaciones[g]:.4f} | "
                      f"Promedio: {self.estadisticas['promedio'][-1]:.4f} | {div_str}")
        
        tiempo_total = time.time() - inicio
        
        print("\n" + "=" * 75)
        print("EVOLUCIÓN COMPLETADA")
        print("=" * 75)
        print(f"Tiempo total: {tiempo_total:.2f} segundos")
        print(f"Nota inicial: {self.mejor_nota_generaciones[0]:.4f}")
        print(f"Nota final: {self.mejor_nota_generaciones[-1]:.4f}")
        print(f"Mejora: {self.mejor_nota_generaciones[-1] - self.mejor_nota_generaciones[0]:.4f} "
              f"({100 * (self.mejor_nota_generaciones[-1] - self.mejor_nota_generaciones[0]):.1f}%)")
        print(f"Diversidad final: {self.estadisticas['diversidad'][-1]:.4f}")
        print("=" * 75 + "\n")
        
        return tiempo_total
    
    def visualizar_resultados(self):
        """Genera gráficos de evolución"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Evolución de fitness
        ax = axes[0, 0]
        gen = np.arange(self.gen_max)
        ax.plot(gen, self.estadisticas['mejor'], 'g-', linewidth=2.5, label='Mejor', marker='o', markersize=4)
        ax.plot(gen, self.estadisticas['promedio'], 'b-', linewidth=2, label='Promedio', alpha=0.7)
        ax.plot(gen, self.estadisticas['peor'], 'r-', linewidth=1.5, label='Peor', alpha=0.6)
        ax.fill_between(gen, self.estadisticas['peor'], self.estadisticas['mejor'], alpha=0.1, color='gray')
        ax.set_xlabel('Generación', fontweight='bold')
        ax.set_ylabel('Fitness', fontweight='bold')
        ax.set_title('Evolución del Fitness', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Gráfico 2: Mejora por generación
        ax = axes[0, 1]
        mejora = np.diff(self.estadisticas['mejor'], prepend=self.estadisticas['mejor'][0])
        colores = ['#2ecc71' if m > 0 else '#e74c3c' for m in mejora]
        ax.bar(gen, mejora, color=colores, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Generación', fontweight='bold')
        ax.set_ylabel('Δ Fitness', fontweight='bold')
        ax.set_title('Cambio de Fitness por Generación', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linewidth=1)
        
        # Gráfico 3: Diversidad
        ax = axes[1, 0]
        ax.plot(gen, self.estadisticas['diversidad'], 'purple', linewidth=2.5, marker='s', markersize=4)
        ax.fill_between(gen, self.estadisticas['diversidad'], alpha=0.3, color='purple')
        ax.set_xlabel('Generación', fontweight='bold')
        ax.set_ylabel('Diversidad Genética', fontweight='bold')
        ax.set_title('Diversidad de la Población', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(self.estadisticas['diversidad']) * 1.1 if self.estadisticas['diversidad'] else 1])
        
        # Gráfico 4: Dispersión (Mejor vs Promedio)
        ax = axes[1, 1]
        dispersión = np.array(self.estadisticas['mejor']) - np.array(self.estadisticas['promedio'])
        ax.fill_between(gen, 0, dispersión, alpha=0.5, color='orange', label='Brecha (Mejor-Promedio)')
        ax.plot(gen, dispersión, 'orange', linewidth=2.5, marker='d', markersize=4)
        ax.set_xlabel('Generación', fontweight='bold')
        ax.set_ylabel('Diferencia de Fitness', fontweight='bold')
        ax.set_title('Selección (Brecha entre Mejor y Promedio)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def analizar_mejor_individuo(self):
        """Analiza el mejor individuo encontrado"""
        from utils import randvec01
        
        if not self.mejores_individuos:
            print("No hay individuos para analizar")
            return
        
        mejor_ca = self.mejores_individuos[-1]
        print("\n" + "=" * 75)
        print("ANÁLISIS DEL MEJOR AUTÓMATA")
        print("=" * 75)
        print(f"Radio: {mejor_ca.r}")
        print(f"Ancho: {mejor_ca.width}")
        print(f"Timesteps: {mejor_ca.timesteps}")
        print(f"Regla (primeros 16 bits): {mejor_ca.rule[:16].astype(int)}")
        print(f"Regla decimal: {int(''.join(mejor_ca.rule.astype(int).astype(str)), 2)}")
        print(f"Fitness alcanzado: {self.mejor_nota_generaciones[-1]:.4f}")
        
        # Prueba con diferentes densidades
        print(f"\n{'Densidad inicial':<20} {'Densidad final':<20} {'Convergencia'}")
        print("-" * 60)
        
        for densidad in [0.2, 0.3, 0.5, 0.7, 0.8]:
            ic = randvec01(1, mejor_ca.width, densidad)[0]
            evol = mejor_ca._run(ic)
            densidad_final = np.mean(evol[-1, :])
            converge = "Sí (0s)" if densidad_final < 0.2 else "Sí (1s)" if densidad_final > 0.8 else "No"
            print(f"{densidad:<20.1f} {densidad_final:<20.4f} {converge}")
        
        print("=" * 75)


if __name__ == "__main__":
    # Ejecutar AG con parámetros predefinidos
    ag = GeneticAlgorithm(
        poblacion=50,
        num_parents=5,
        gen_max=50,
        num_CI=20,
        ancho=101,
        r=3,
        p_mutacion=0.05
    )
    
    tiempo = ag.evolucionar(verbose=True)
    ag.visualizar_resultados()
    ag.analizar_mejor_individuo()

