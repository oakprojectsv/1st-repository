"""
network.py
Red neuronal multicapa (feedforward), implementada desde cero con NumPy.

Arquitectura: lista de tamaños de capa, ej. [2, 4, 1] significa:
  - 2 neuronas de entrada
  - 4 neuronas en la capa oculta
  - 1 neurona de salida

Todo el forward y backward pass está escrito explícitamente para que
veas la conexión directa con las fórmulas matemáticas.
"""

import numpy as np
from activations import sigmoid, sigmoid_derivative


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.5, seed=42):
        """
        layer_sizes: lista, ej. [2, 4, 1] -> 2 entradas, 1 capa oculta de 4, 1 salida
        """
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.lr = learning_rate

        self.weights = []   # W[l] tiene forma (n_l, n_{l-1})
        self.biases = []    # b[l] tiene forma (n_l,)

        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            # Inicialización pequeña aleatoria (rompe simetría entre neuronas)
            W = rng.normal(0, 1, size=(n_out, n_in)) * np.sqrt(1 / n_in)
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # FORWARD PROPAGATION
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x: vector de entrada, shape (n_inputs,)
        Devuelve: (activaciones de cada capa, valores z de cada capa)
        activations[0] = x (la entrada, tratada como "capa 0")
        """
        activations = [x]
        zs = []

        a = x
        for W, b in zip(self.weights, self.biases):
            z = W @ a + b          # z = W*a + b  (multiplicación matriz-vector)
            a = sigmoid(z)         # activación no lineal
            zs.append(z)
            activations.append(a)

        return activations, zs

    def predict(self, x):
        activations, _ = self.forward(x)
        return activations[-1]     # la salida de la última capa

    # ------------------------------------------------------------------
    # BACKPROPAGATION
    # ------------------------------------------------------------------
    def backward(self, activations, zs, y_true):
        """
        Calcula los gradientes de la pérdida respecto a cada W y b,
        usando la regla de la cadena, capa por capa, DE ATRÁS HACIA ADELANTE.

        Pérdida usada: MSE, L = (1/2)*sum((y_true - y_pred)^2)
        (el 1/2 es solo para que la derivada quede más limpia: -( y_true - y_pred))
        """
        n_layers = len(self.weights)
        grad_w = [None] * n_layers
        grad_b = [None] * n_layers

        y_pred = activations[-1]

        # --- Paso 1: error en la última capa ---
        # dL/da_L = -(y_true - y_pred)          (derivada de MSE)
        # dL/dz_L = dL/da_L * da_L/dz_L = dL/da_L * sigmoid'(a_L)
        dL_da = -(y_true - y_pred)
        delta = dL_da * sigmoid_derivative(y_pred)   # "delta" = error en z de esta capa

        grad_w[-1] = np.outer(delta, activations[-2])   # dL/dW = delta * a_prev^T
        grad_b[-1] = delta

        # --- Paso 2: propagar el error hacia atrás por las capas ocultas ---
        for l in range(n_layers - 2, -1, -1):
            # Cuánto error "le llega" a la capa l desde la capa l+1:
            # delta_l = (W_{l+1}^T @ delta_{l+1}) * f'(a_l)
            delta = (self.weights[l + 1].T @ delta) * sigmoid_derivative(activations[l + 1])
            grad_w[l] = np.outer(delta, activations[l])
            grad_b[l] = delta

        return grad_w, grad_b

    # ------------------------------------------------------------------
    # ACTUALIZACIÓN DE PESOS (gradiente descendente)
    # ------------------------------------------------------------------
    def update_weights(self, grad_w, grad_b):
        for l in range(len(self.weights)):
            # w = w - lr * dL/dw   (descenso en la dirección opuesta al gradiente)
            self.weights[l] -= self.lr * grad_w[l]
            self.biases[l] -= self.lr * grad_b[l]

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------
    def train_step(self, x, y_true):
        activations, zs = self.forward(x)
        grad_w, grad_b = self.backward(activations, zs, y_true)
        self.update_weights(grad_w, grad_b)

        y_pred = activations[-1]
        loss = 0.5 * np.sum((y_true - y_pred) ** 2)
        return loss

    def fit(self, X, Y, epochs=5000, verbose_every=500):
        """
        X: lista/array de vectores de entrada
        Y: lista/array de salidas esperadas (una por cada fila de X)
        """
        history = []
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for x, y in zip(X, Y):
                total_loss += self.train_step(np.array(x, dtype=float), np.array(y, dtype=float))
            avg_loss = total_loss / len(X)
            history.append(avg_loss)

            if verbose_every and epoch % verbose_every == 0:
                print(f"Época {epoch:5d} | pérdida promedio = {avg_loss:.6f}")

        return history
