"""
gui.py
Interfaz gráfica (Tkinter) de la calculadora de vectores en R^n.
Si los vectores son de dimensión 2 o 3, se puede visualizar en un
gráfico 3D (matplotlib embebido en la ventana).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import vector_ops as vo

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necesario para proyección 3d)


class VectorCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Vectores en R^n")
        self.root.geometry("980x640")

        self._build_layout()

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    def _build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 10))

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        # --- Entrada de vectores ---
        ttk.Label(left, text="Calculadora de Vectores en R^n", font=("TkDefaultFont", 13, "bold")).pack(anchor="w", pady=(0, 10))

        ttk.Label(left, text="Vector A (separa componentes con comas):").pack(anchor="w")
        self.entry_a = ttk.Entry(left, width=40)
        self.entry_a.insert(0, "4, -5, 9")
        self.entry_a.pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Vector B (separa componentes con comas):").pack(anchor="w")
        self.entry_b = ttk.Entry(left, width=40)
        self.entry_b.insert(0, "6, -2, 4")
        self.entry_b.pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Escalar k (para A·k o B·k):").pack(anchor="w")
        self.entry_k = ttk.Entry(left, width=10)
        self.entry_k.insert(0, "2")
        self.entry_k.pack(anchor="w", pady=(0, 12))

        # --- Botones de operaciones ---
        ops_frame = ttk.LabelFrame(left, text="Operaciones", padding=8)
        ops_frame.pack(fill="x", pady=(0, 12))

        ops = [
            ("A + B", self.op_add),
            ("A - B", self.op_subtract),
            ("A · B (punto)", self.op_dot),
            ("A x B (cruz, solo R^3)", self.op_cross),
            ("|A|  y  |B|", self.op_magnitude),
            ("Vector unitario de A", self.op_unit_a),
            ("Ángulo entre A y B", self.op_angle),
            ("A × k (escalar)", self.op_scalar_a),
        ]
        for i, (label, cmd) in enumerate(ops):
            ttk.Button(ops_frame, text=label, command=cmd).grid(
                row=i // 2, column=i % 2, padx=4, pady=4, sticky="ew"
            )
        ops_frame.columnconfigure(0, weight=1)
        ops_frame.columnconfigure(1, weight=1)

        # --- Sección de independencia lineal (n vectores) ---
        indep_frame = ttk.LabelFrame(left, text="Dependencia / independencia lineal", padding=8)
        indep_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(indep_frame, text="Un vector por línea (Enter para agregar):").pack(anchor="w")
        self.text_vectors = tk.Text(indep_frame, width=40, height=5)
        self.text_vectors.insert("1.0", "0, 2, 4\n1, -2, 1\n1, 4, 3")
        self.text_vectors.pack(pady=(4, 6))

        btn_row = ttk.Frame(indep_frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Verificar independencia", command=self.op_independence).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Determinante", command=self.op_determinant).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Graficar estos vectores", command=self.plot_from_textbox).pack(side="left", padx=2)

        # --- Sección de cambio de base ---
        basis_frame = ttk.LabelFrame(left, text="Coordenadas respecto a una base", padding=8)
        basis_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(
            basis_frame,
            text="Usa los vectores de la caja de arriba como base.\nVector a expresar en esa base:"
        ).pack(anchor="w")
        self.entry_basis_vector = ttk.Entry(basis_frame, width=40)
        self.entry_basis_vector.insert(0, "1, 4, 2")
        self.entry_basis_vector.pack(anchor="w", pady=(4, 6))

        ttk.Button(
            basis_frame, text="Calcular coordenadas en esa base", command=self.op_coordinates_in_basis
        ).pack(anchor="w")

        # --- Resultado (readonly Entry, tal como vimos en la guía) ---
        ttk.Label(left, text="\nResultado:", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.result_box = tk.Text(left, width=45, height=6, state="disabled", bg="#f4f4f4")
        self.result_box.pack(pady=(4, 0))

        # --- Panel derecho: gráfico 3D ---
        ttk.Label(right, text="Visualización (solo para vectores en R² o R³)",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.figure = Figure(figsize=(5.5, 5.5), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self._reset_plot()

        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ttk.Button(right, text="Graficar A y B", command=self.plot_ab).pack(pady=6)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_result(self, text):
        self.result_box.config(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", text)
        self.result_box.config(state="disabled")

    def _get_vector(self, entry_widget, label):
        try:
            return vo.parse_vector(entry_widget.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Vector {label}: {e}")
            return None

    def _get_vectors_from_textbox(self):
        raw_lines = self.text_vectors.get("1.0", "end").strip().splitlines()
        vectors = []
        for i, line in enumerate(raw_lines, start=1):
            if line.strip() == "":
                continue
            try:
                vectors.append(vo.parse_vector(line))
            except ValueError as e:
                messagebox.showerror("Error", f"Línea {i}: {e}")
                return None
        if len(vectors) < 1:
            messagebox.showerror("Error", "Agrega al menos un vector.")
            return None
        return vectors

    def _fmt(self, arr):
        return "(" + ", ".join(f"{x:.4g}" for x in np.atleast_1d(arr)) + ")"

    # ------------------------------------------------------------------
    # Operaciones (callbacks de los botones)
    # ------------------------------------------------------------------
    def op_add(self):
        a, b = self._get_vector(self.entry_a, "A"), self._get_vector(self.entry_b, "B")
        if a is None or b is None:
            return
        try:
            result = vo.add(a, b)
            self._set_result(f"A + B = {self._fmt(result)}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def op_subtract(self):
        a, b = self._get_vector(self.entry_a, "A"), self._get_vector(self.entry_b, "B")
        if a is None or b is None:
            return
        try:
            result = vo.subtract(b, a)  # AB = B - A, convención de la guía
            self._set_result(f"B - A (vector AB) = {self._fmt(result)}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def op_dot(self):
        a, b = self._get_vector(self.entry_a, "A"), self._get_vector(self.entry_b, "B")
        if a is None or b is None:
            return
        try:
            self._set_result(f"A · B = {vo.dot_product(a, b):.6g}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def op_cross(self):
        a, b = self._get_vector(self.entry_a, "A"), self._get_vector(self.entry_b, "B")
        if a is None or b is None:
            return
        try:
            result = vo.cross_product(a, b)
            self._set_result(f"A x B = {self._fmt(result)}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def op_magnitude(self):
        a, b = self._get_vector(self.entry_a, "A"), self._get_vector(self.entry_b, "B")
        if a is None or b is None:
            return
        self._set_result(f"|A| = {vo.magnitude(a):.6g}\n|B| = {vo.magnitude(b):.6g}")

    def op_unit_a(self):
        a = self._get_vector(self.entry_a, "A")
        if a is None:
            return
        try:
            self._set_result(f"Vector unitario de A = {self._fmt(vo.unit_vector(a))}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def op_angle(self):
        a, b = self._get_vector(self.entry_a, "A"), self._get_vector(self.entry_b, "B")
        if a is None or b is None:
            return
        try:
            self._set_result(f"Ángulo entre A y B = {vo.angle_between(a, b):.4f}°")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def op_scalar_a(self):
        a = self._get_vector(self.entry_a, "A")
        if a is None:
            return
        try:
            k = float(self.entry_k.get())
        except ValueError:
            messagebox.showerror("Error", "El escalar k debe ser un número.")
            return
        self._set_result(f"A × {k} = {self._fmt(vo.scalar_multiply(a, k))}")

    def op_independence(self):
        vectors = self._get_vectors_from_textbox()
        if vectors is None:
            return
        try:
            is_indep, rank, detalle = vo.linear_independence(vectors)
            veredicto = "LINEALMENTE INDEPENDIENTES" if is_indep else "LINEALMENTE DEPENDIENTES"
            self._set_result(f"{veredicto}\n{detalle}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def op_determinant(self):
        vectors = self._get_vectors_from_textbox()
        if vectors is None:
            return
        try:
            det = vo.determinant(vectors)
            self._set_result(f"Determinante = {det:.6g}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def op_coordinates_in_basis(self):
        basis_vectors = self._get_vectors_from_textbox()
        if basis_vectors is None:
            return
        try:
            v = vo.parse_vector(self.entry_basis_vector.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Vector a expresar: {e}")
            return
        try:
            coords = vo.coordinates_in_basis(v, basis_vectors)
            self._set_result(
                "Coordenadas en la base dada:\n"
                f"{self._fmt(coords)}\n\n"
                "(en el mismo orden en que escribiste los vectores base)"
            )
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------------
    # Visualización 3D
    # ------------------------------------------------------------------
    def _reset_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Vectores desde el origen")

    def _draw_vector(self, v, color, label):
        v3 = np.zeros(3)
        v3[: min(3, len(v))] = v[: min(3, len(v))]
        self.ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color=color, label=label, linewidth=2)

    def plot_ab(self):
        a, b = self._get_vector(self.entry_a, "A"), self._get_vector(self.entry_b, "B")
        if a is None or b is None:
            return
        if len(a) > 3 or len(b) > 3:
            messagebox.showwarning(
                "Aviso",
                "Solo se pueden graficar las primeras 3 componentes; "
                "el vector tiene más de 3 dimensiones (R^n con n>3)."
            )
        self._reset_plot()
        self._draw_vector(a, "tab:blue", "A")
        self._draw_vector(b, "tab:red", "B")
        self._autoscale([a, b])
        self.ax.legend()
        self.canvas.draw()

    def plot_from_textbox(self):
        vectors = self._get_vectors_from_textbox()
        if vectors is None:
            return
        self._reset_plot()
        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
        for i, v in enumerate(vectors):
            self._draw_vector(v, colors[i % len(colors)], f"v{i+1}")
        self._autoscale(vectors)
        self.ax.legend()
        self.canvas.draw()

    def _autoscale(self, vectors):
        max_val = max(1.0, max(np.max(np.abs(v[:3])) for v in vectors))
        lim = max_val * 1.2
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        self.ax.set_zlim([-lim, lim])


def main():
    root = tk.Tk()
    app = VectorCalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
