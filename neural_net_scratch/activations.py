"""
activations.py
Funciones de activación y sus derivadas.

Cada función de activación necesita su derivada porque backpropagation
usa la regla de la cadena: para saber cómo un cambio en z afecta la
pérdida final, necesitamos df/dz en cada capa.
"""

import numpy as np


def sigmoid(z):
    """Aplasta cualquier real al rango (0, 1)."""
    # Recorte (clip) para evitar overflow numérico con exponentes grandes
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """
    Derivada de sigmoid, EXPRESADA EN TÉRMINOS DE LA SALIDA a = sigmoid(z).
    d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z)) = a * (1 - a)
    Esto es una optimización: como ya calculamos 'a' en forward pass,
    no hace falta recalcular sigmoid(z) de nuevo.
    """
    return a * (1 - a)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    """La derivada de ReLU es 1 donde z>0, y 0 donde z<=0 (no diferenciable en 0,
    mundo real 0)."""
    return (z > 0).astype(float)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
}
