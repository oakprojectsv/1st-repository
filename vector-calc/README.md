# Calculadora de Vectores en R^n

Calculadora de vectores con interfaz gráfica (Tkinter), que soporta **cualquier
dimensión (R^n)** para las operaciones numéricas, y **visualización 3D**
(matplotlib) cuando los vectores tienen 2 o 3 componentes.

## Estructura del proyecto

```
vector_calc/
├── main.py         # Punto de entrada
├── gui.py          # Interfaz gráfica (Tkinter + matplotlib embebido)
├── vector_ops.py   # Lógica matemática pura (sin dependencia de la GUI)
└── requirements.txt
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

En Omarchy/Arch, si `tkinter` no está disponible:
```bash
sudo pacman -S tk
```

## Uso

```bash
python main.py
```

## Funcionalidades

- Operaciones entre dos vectores A y B: suma, resta, producto punto,
  producto cruz (R^3), magnitud, vector unitario, ángulo entre vectores,
  multiplicación por escalar.
- Verificación de dependencia/independencia lineal para **n vectores**
  de cualquier dimensión, usando el rango de la matriz.
- Cálculo de determinante (cuando aplica, matriz cuadrada).
- Visualización 3D de vectores desde el origen (si tienen ≤3 componentes;
  para R^n con n>3 solo se grafican las primeras 3 componentes).

## Próximos pasos sugeridos

- Cálculo de coordenadas de un vector respecto a una base arbitraria
  (ya implementado en `vector_ops.coordinates_in_basis`, falta exponerlo en la GUI)
- Cambio de base completo (base A → base B)
- Exportar resultados a un archivo
