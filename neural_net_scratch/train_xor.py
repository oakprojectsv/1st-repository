"""
train_xor.py
Entrena la red neuronal para resolver XOR: el ejemplo clásico que
demuestra por qué se necesita al menos una capa oculta (no es
linealmente separable).

Tabla de verdad de XOR:
  (0,0) -> 0
  (0,1) -> 1
  (1,0) -> 1
  (1,1) -> 0
"""

import numpy as np
from network import NeuralNetwork

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
Y = [
    [0],
    [1],
    [1],
    [0],
]

# Arquitectura: 2 entradas -> 4 neuronas ocultas -> 1 salida
net = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=0.5, seed=42)

print("Entrenando red neuronal para resolver XOR...\n")
history = net.fit(X, Y, epochs=5000, verbose_every=1000)

print("\nResultados finales:")
for x, y in zip(X, Y):
    pred = net.predict(np.array(x, dtype=float))
    print(f"entrada={x} | esperado={y[0]} | predicción={pred[0]:.4f} | redondeado={round(pred[0])}")
