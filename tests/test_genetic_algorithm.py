import numpy as np
import pytest
from models import GeneticAlgorithm, CellularAutomaton



def test_ga_initial_population_size():
    ga = GeneticAlgorithm(poblacion=20, gen_max=1)
    assert len(ga.poblacion_ca) == 20


def test_ga_rule_length_consistency():
    ga = GeneticAlgorithm(r=2, poblacion=10)
    expected = 2 ** (2 * ga.r + 1)

    for ca in ga.poblacion_ca:
        assert len(ca.rule) == expected



def test_ga_evaluation_shape():
    ga = GeneticAlgorithm(poblacion=5, gen_max=1)

    A = np.random.randint(0, 2, (ga.num_CI, ga.ancho))
    Cf = np.random.randint(0, 2, (ga.num_CI, ga.ancho))

    fitness = ga._evaluar_poblacion(A, Cf)

    assert fitness.shape[0] == ga.poblacion
    assert np.all((fitness >= 0) & (fitness <= 1))


def test_individual_evaluation_range():
    ga = GeneticAlgorithm(poblacion=1, gen_max=1)

    ca = ga.poblacion_ca[0]
    A = np.random.randint(0, 2, (1, ga.ancho))
    Cf = np.random.randint(0, 2, (1, ga.ancho))

    score = ga._evaluate_individual(ca, A, Cf)

    assert 0 <= score <= 1



def test_crossover_creates_valid_child():
    ga = GeneticAlgorithm(r=2)

    p1 = ga.poblacion_ca[0]
    p2 = ga.poblacion_ca[1]

    child = ga._crossover(p1, p2)

    assert isinstance(child, CellularAutomaton)
    assert len(child.rule) == len(p1.rule)
    assert child.r == ga.r



def test_mutation_changes_rule():
    ga = GeneticAlgorithm(poblacion=1, r=2)

    ca = ga.poblacion_ca[0]
    original = ca.rule.copy()

    ga._mutate(ca)

    assert not np.array_equal(original, ca.rule)


def test_mutation_bounds():
    ga = GeneticAlgorithm(r=2)

    ca = ga.poblacion_ca[0]
    ga._mutate(ca)

    assert np.all((ca.rule == 0) | (ca.rule == 1))



def test_selection_returns_top_parents():
    ga = GeneticAlgorithm(poblacion=10, num_parents=3, gen_max=1)

    A = np.random.randint(0, 2, (ga.num_CI, ga.ancho))
    Cf = np.random.randint(0, 2, (ga.num_CI, ga.ancho))

    fitness = ga._evaluar_poblacion(A, Cf)

    idx = np.argsort(-fitness)
    best = fitness[idx[0]]

    assert best == np.max(fitness)



def test_diversity_non_negative():
    ga = GeneticAlgorithm(poblacion=10)

    div = ga._calcular_diversidad()

    assert div >= 0



def test_evolution_runs_without_error():
    ga = GeneticAlgorithm(poblacion=10, gen_max=2)

    ga.evolucionar(verbose=False)

    assert len(ga.mejor_nota_generaciones) == ga.gen_max



def test_statistics_are_recorded():
    ga = GeneticAlgorithm(poblacion=10, gen_max=2)

    ga.evolucionar(verbose=False)

    assert len(ga.estadisticas['mejor']) == ga.gen_max
    assert len(ga.estadisticas['promedio']) == ga.gen_max
    assert len(ga.estadisticas['diversidad']) == ga.gen_max