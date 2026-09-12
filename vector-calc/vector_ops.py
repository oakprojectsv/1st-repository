"""
vector_ops.py
Operaciones vectoriales en R^n usando numpy.
Toda la lógica matemática vive aquí, separada de la interfaz gráfica (GUI).
"""

import numpy as np


def parse_vector(text):
    """Convierte un string tipo '1, 2, 3' o '1 2 3' en un array de numpy."""
    text = text.replace(",", " ")
    parts = [p for p in text.split() if p.strip() != ""]
    if not parts:
        raise ValueError("El vector está vacío.")
    return np.array([float(p) for p in parts], dtype=float)


def check_same_dimension(a, b):
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Los vectores tienen dimensiones distintas: R^{a.shape[0]} y R^{b.shape[0]}."
        )


def add(a, b):
    check_same_dimension(a, b)
    return a + b


def subtract(a, b):
    check_same_dimension(a, b)
    return a - b


def scalar_multiply(a, k):
    return a * k


def dot_product(a, b):
    check_same_dimension(a, b)
    return float(np.dot(a, b))


def magnitude(a):
    return float(np.linalg.norm(a))


def unit_vector(a):
    mag = magnitude(a)
    if mag == 0:
        raise ValueError("No se puede normalizar el vector cero.")
    return a / mag


def angle_between(a, b, degrees=True):
    check_same_dimension(a, b)
    mag_a, mag_b = magnitude(a), magnitude(b)
    if mag_a == 0 or mag_b == 0:
        raise ValueError("No se puede calcular el ángulo con un vector cero.")
    cos_theta = np.clip(dot_product(a, b) / (mag_a * mag_b), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta)) if degrees else float(theta)


def cross_product(a, b):
    """Producto cruz, solo definido de forma clásica en R^3."""
    if a.shape[0] != 3 or b.shape[0] != 3:
        raise ValueError("El producto cruz clásico solo está definido en R^3.")
    return np.cross(a, b)


def linear_independence(vectors):
    """
    Recibe una lista de vectores (arrays) y determina si son
    linealmente independientes, usando el rango de la matriz.
    Devuelve (es_independiente: bool, rango: int, detalle: str)
    """
    dims = {v.shape[0] for v in vectors}
    if len(dims) > 1:
        raise ValueError("Todos los vectores deben tener la misma dimensión.")

    matrix = np.array(vectors)
    rank = np.linalg.matrix_rank(matrix)
    n_vectors = len(vectors)

    detalle = f"Rango de la matriz: {rank} | Número de vectores: {n_vectors}"

    if rank == n_vectors:
        return True, rank, detalle
    else:
        return False, rank, detalle


def determinant(vectors):
    """Determinante, solo si la matriz formada por los vectores es cuadrada."""
    matrix = np.array(vectors)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "El determinante solo se puede calcular con una matriz cuadrada "
            "(mismo número de vectores que de componentes)."
        )
    return float(np.linalg.det(matrix))


def coordinates_in_basis(vector, basis_vectors):
    """
    Encuentra las coordenadas de `vector` respecto a una base dada
    (lista de vectores linealmente independientes), resolviendo el
    sistema de ecuaciones correspondiente.
    """
    is_indep, rank, _ = linear_independence(basis_vectors)
    if not is_indep:
        raise ValueError("El conjunto dado no es una base (los vectores no son "
                          "linealmente independientes).")

    matrix = np.array(basis_vectors).T  # cada vector de la base como columna
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("El número de vectores en la base debe igualar la dimensión "
                          "del espacio para resolver un sistema único.")

    coords = np.linalg.solve(matrix, vector)
    return coords
