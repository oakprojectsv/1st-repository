# Red Neuronal desde Cero (sin frameworks)

Implementación completa de una red neuronal feedforward usando solo
NumPy — para entender los fundamentos matemáticos (forward propagation,
backpropagation, gradiente descendente) antes de usar PyTorch/TensorFlow.

## Estructura

```
neural_net_scratch/
├── activations.py   # sigmoid, ReLU y sus derivadas
├── neuron.py         # una sola neurona (educativo, aislado)
├── network.py         # la red completa: forward + backward + entrenamiento
└── train_xor.py       # ejemplo de entrenamiento: resolver XOR
```

## Ejecutar

```bash
python -m venv venv
source venv/bin/activate
pip install numpy
python train_xor.py
```

## Qué demuestra este proyecto

XOR **no es linealmente separable** — ninguna línea recta separa las
entradas que dan 0 de las que dan 1. Por eso una sola neurona (o red
sin capas ocultas) no puede resolverlo, sin importar qué pesos tenga.

La red usada aquí (`[2, 4, 1]`: 2 entradas, 4 neuronas ocultas, 1 salida)
sí lo resuelve, porque la capa oculta transforma el espacio de entrada
a uno donde el problema sí se vuelve separable.

## Resultado esperado

Después de 5000 épocas de entrenamiento, la pérdida (MSE) baja de
~0.4 a menos de 0.001, y las predicciones finales coinciden con la
tabla de verdad de XOR:

| entrada | esperado | predicción |
|---|---|---|
| (0,0) | 0 | ~0.02 |
| (0,1) | 1 | ~0.97 |
| (1,0) | 1 | ~0.98 |
| (1,1) | 0 | ~0.02 |

## Matemática implementada

- **Forward pass:** `z = W @ a + b`, `a = sigmoid(z)` en cada capa
- **Pérdida:** Error Cuadrático Medio (MSE)
- **Backward pass:** regla de la cadena aplicada capa por capa (backpropagation)
- **Optimización:** descenso de gradiente (`w -= learning_rate * dL/dw`)

## Próximos pasos sugeridos

- Agregar más funciones de activación (ReLU ya está implementada en `activations.py`, falta usarla en `network.py`)
- Entrenar con mini-batches en vez de un ejemplo a la vez
- Probar con un dataset real (ej. clasificación de dígitos MNIST)
- Comparar el resultado con la misma red hecha en PyTorch (para ver qué tanto simplifica el framework)
