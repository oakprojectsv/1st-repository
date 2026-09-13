"""
neuron.py
La pieza más básica: una sola neurona.

z = w . x + b        (producto punto + bias)
a = f(z)              (activación)

Este archivo es solo EDUCATIVO — para que veas la operación de una
neurona aislada antes de pasar a capas completas con matrices.
"""

import numpy as np
from activations import sigmoid


class Neuron:
    def __init__(self, n_inputs):
        # Pesos iniciales aleatorios pequeños (romper simetría)
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = 0.0

    def forward(self, x):
        z = np.dot(self.w, x) + self.b   # exactamente w . x + b
        a = sigmoid(z)
        return a, z


if __name__ == "__main__":
    # Prueba rápida: una neurona con 3 entradas
    neuron = Neuron(n_inputs=3)
    x = np.array([1.0, 0.5, -1.0])
    a, z = neuron.forward(x)
    print(f"pesos w = {neuron.w}")
    print(f"z = w·x + b = {z:.4f}")
    print(f"a = sigmoid(z) = {a:.4f}")
