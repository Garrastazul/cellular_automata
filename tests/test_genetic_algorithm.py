import pytest
import numpy as np
from models import CellularAutomaton, binary_to_int, int_to_binary_rule


# =========================
# 🔁 CONVERSIÓN DE REGLAS
# =========================

def test_rule_roundtrip_deterministic():
    for r in range(1, 4):
        for rule_num in [0, 1, 2, 10, 50, 127, 255]:
            rule = int_to_binary_rule(rule_num, r)
            assert binary_to_int(rule) == rule_num


def test_rule_length():
    for r in range(1, 5):
        rule = int_to_binary_rule(10, r)
        assert len(rule) == 2 ** (2 * r + 1)


def test_binary_inversion_consistency():
    for r in range(1, 4):
        rule_num = np.random.randint(2 ** (2 * r + 1))
        rule = int_to_binary_rule(rule_num, r)
        assert np.array_equal(rule, int_to_binary_rule(binary_to_int(rule), r))


# =========================
# 🧠 AUTÓMATA CELULAR
# =========================

def test_ca_output_shape():
    r = 2
    width = 20
    rule = np.random.randint(0, 2, 2 ** (2 * r + 1))

    ca = CellularAutomaton(rule, r=r, width=width, timesteps=15)
    initial = np.random.randint(0, 2, width)

    evolution = ca._run(initial)

    assert evolution.shape == (15, width)


def test_ca_deterministic():
    r = 2
    width = 20
    rule = np.random.randint(0, 2, 2 ** (2 * r + 1))

    ca = CellularAutomaton(rule, r=r, width=width, timesteps=10)
    initial = np.random.randint(0, 2, width)

    e1 = ca._run(initial)
    e2 = ca._run(initial)

    assert np.array_equal(e1, e2)


def test_ca_constant_rule():
    r = 1
    width = 15
    rule = np.zeros(2 ** (2 * r + 1), dtype=int)

    ca = CellularAutomaton(rule, r=r, width=width, timesteps=10)
    initial = np.zeros(width, dtype=int)

    evolution = ca._run(initial)

    assert np.all(evolution == 0)


def test_ca_single_active_cell():
    r = 1
    width = 15

    rule = np.zeros(2 ** (2 * r + 1), dtype=int)
    rule[len(rule)//2] = 1  # regla simple de propagación

    ca = CellularAutomaton(rule, r=r, width=width, timesteps=5)

    initial = np.zeros(width, dtype=int)
    initial[width // 2] = 1

    evolution = ca._run(initial)

    assert evolution.shape[0] == 5


def test_ca_mutation_changes_rule():
    r = 2
    rule = np.ones(2 ** (2 * r + 1), dtype=int)

    ca = CellularAutomaton(rule, r=r)

    mutated = ca._mutate(mutation_rate=1.0)

    assert not np.array_equal(rule, mutated)