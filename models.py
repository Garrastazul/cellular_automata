import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit


def binary_to_int(bin_vec):
    return sum(bit * (2 ** i) for i, bit in enumerate(reversed(bin_vec)))

def int_to_binary_rule(rule_number: int, r):
    bit_length = 2 ** (2 * r + 1)
    rule_binary = np.array([(int(rule_number) >> (bit_length - 1 - i)) & 1 for i in range(bit_length)])
    return rule_binary

@njit(cache=True)
def _run_ca(rule, r, timesteps, AN, initial_state):
    M = np.zeros((timesteps, AN + 2 * r))
    M[0, :r] = initial_state[AN - r:]
    M[0, r:AN + r] = initial_state
    M[0, AN + r:] = initial_state[:r]

    exp = np.zeros(2 * r + 1)
    for k in range(2 * r + 1):
        exp[k] = 2.0 ** (2 * r - k)

    lR = len(rule)
    for i in range(1, timesteps):
        for j in range(r, AN + r):
            indice = 0
            for k in range(2 * r + 1):
                indice += int(M[i-1, j - r + k] * exp[k])
            M[i, j] = rule[lR - 1 - indice]
        M[i, :r] = M[i, AN:AN + r]
        M[i, AN + r:] = M[i, r:2 * r]

    return M[:, r:AN + r]


class CellularAutomaton:
    """Clase pura de simulación: Representa el modelo y su lógica de transición"""
    
    def __init__(self, rule, r=3, width=101, timesteps=None):
        self.rule = rule.astype(int)
        self.r = r
        self.width = width
        self.timesteps = timesteps if timesteps else 2 * width
        self.trajectory = None
    
    def run(self, initial_state):
        """Ejecuta la simulación del autómata temporalmente"""
        return _run_ca(self.rule, self.r, self.timesteps, len(initial_state),
                       initial_state.astype(np.float64))
    
    def copy(self):
        return CellularAutomaton(self.rule.copy(), self.r, self.width, self.timesteps)
    
    def __repr__(self):
        rule_str = ''.join(self.rule.astype(int).astype(str))[:8] + '...'
        return f"CA(r={self.r}, rule_bits={len(self.rule)}, {rule_str})"
    
    def _save_trajectory(self, actual_state):
        if self.trajectory == None:
            self.trajectory = []
        self.trajectory.append(actual_state)




class GeneticAlgorithm:
    """Clase de orquestación: Maneja la evolución, selección y operadores genéticos"""
    
    def __init__(self, poblacion=50, num_parents=5, gen_max=50,
                 num_CI=20, ancho=101, r=3, p_mutacion=0.05):
        from utils import randvec01
        
        self.poblacion = poblacion
        self.num_parents = num_parents
        self.gen_max = gen_max
        self.num_CI = num_CI
        self.ancho = ancho
        self.r = r
        self.p_mutacion = p_mutacion
        self.lR = 2 ** (2 * r + 1)
        
        self.poblacion_ca = [
            CellularAutomaton(randvec01(1, self.lR, 0.5)[0], r=r, width=ancho)
            for _ in range(poblacion)
        ]
        
        self.mejor_nota_generaciones = np.zeros(gen_max)
        self.mejores_individuos = []
        self.estadisticas = {'mejor': [], 'promedio': [], 'peor': [], 'diversidad': []}

    def _evaluate_individual(self, ca, initial_conditions, objectives):
        """Calcula el fitness de un individuo específico"""
        num_CI = len(initial_conditions)
        coincidencias = sum(
            np.sum(ca.run(initial_conditions[i, :])[-1, :] == objectives[i, :])
            for i in range(num_CI)
        )
        return coincidencias / (num_CI * ca.width)

    def _crossover(self, parent1, parent2):
        """Combina dos autómatas para crear un descendiente"""
        point = np.random.randint(0, self.lR)
        child_rule = np.concatenate([parent1.rule[:point], parent2.rule[point:]])
        return CellularAutomaton(child_rule, self.r, self.ancho)

    def _mutate(self, ca):
        """Aplica mutación a la regla de un autómata"""
        mutations = np.random.rand(self.lR) < self.p_mutacion
        ca.rule = np.mod(ca.rule + mutations, 2)

    def _evaluar_poblacion(self, A, Cf):
        return np.array([self._evaluate_individual(ca, A, Cf) for ca in self.poblacion_ca])

    def _calcular_diversidad(self):
        if len(self.poblacion_ca) < 2: return 0.0
        diferencias = 0
        total_comp = 0
        for i in range(min(len(self.poblacion_ca), 10)):
            for j in range(i + 1, min(len(self.poblacion_ca), 10)):
                diferencias += np.sum(self.poblacion_ca[i].rule != self.poblacion_ca[j].rule)
                total_comp += self.lR
        return diferencias / total_comp if total_comp > 0 else 0.0

    def evolucionar(self, verbose=True):
        from utils import CondicionesFinales
        print("=" * 75 + "\nALGORITMO GENÉTICO\n" + "=" * 75)
        inicio = time.time()
        
        for g in range(self.gen_max):
            A, Cf = CondicionesFinales(self.num_CI, self.ancho)
            notas = self._evaluar_poblacion(A, Cf)
            
            self.mejor_nota_generaciones[g] = np.max(notas)
            self.estadisticas['mejor'].append(np.max(notas))
            self.estadisticas['promedio'].append(np.mean(notas))
            self.estadisticas['peor'].append(np.min(notas))
            self.estadisticas['diversidad'].append(self._calcular_diversidad())
            
            indices = np.argsort(-notas)
            padres = [self.poblacion_ca[i].copy() for i in indices[:self.num_parents]]
            
            if g == 0 or self.mejor_nota_generaciones[g] > self.mejor_nota_generaciones[g - 1]:
                self.mejores_individuos.append(padres[0].copy())
            
            nueva_poblacion = padres.copy()
            for _ in range(self.poblacion - self.num_parents):
                p1, p2 = np.random.choice(padres, 2, replace=False)
                hijo = self._crossover(p1, p2)
                self._mutate(hijo)
                nueva_poblacion.append(hijo)
            
            self.poblacion_ca = nueva_poblacion
            
            if verbose and ((g + 1) % 2 == 0 or g == 0):
                print(f"Gen {g+1:3d} | Mejor: {notas.max():.4f} | Div: {self.estadisticas['diversidad'][-1]:.3f}")
        
        return time.time() - inicio

    def visualizar_resultados(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(self.estadisticas['mejor'], label='Mejor')
        axes[0].plot(self.estadisticas['promedio'], label='Promedio')
        axes[0].set_title('Evolución Fitness')
        axes[0].legend()
        
        axes[1].plot(self.estadisticas['diversidad'], color='purple')
        axes[1].set_title('Diversidad Genética')
        plt.show()

    def analizar_mejor_individuo(self):
        from utils import randvec01
        if not self.mejores_individuos: return
        mejor = self.mejores_individuos[-1]
        print(f"\nMejor Fitness: {self.mejor_nota_generaciones[-1]:.4f}")
        print(f"Regla: {mejor.rule[:16].astype(int)}...")

if __name__ == "__main__":
    ag = GeneticAlgorithm(poblacion=100, gen_max=100)
    ag.evolucionar()
    ag.visualizar_resultados()
    ag.analizar_mejor_individuo()