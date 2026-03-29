import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def safe_compile(expr: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Safely compiles a mathematical expression string into a NumPy-vectorized function.
    
    Supports: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, log (np.log), log10, exp,
    sqrt, abs, pi, e, and NumPy array operations.
    
    Args:
        expr: String expression in terms of 'x', e.g., 'sin(x) + cos(x) * x**2'.
    
    Returns:
        Callable that accepts np.ndarray x and returns np.ndarray y.
    
    Raises:
        SyntaxError or other eval errors for invalid expressions.
    """
    safe_dict = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'asin': np.arcsin,
        'acos': np.arccos,
        'atan': np.arctan,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'log': np.log,
        'log10': np.log10,
        'exp': np.exp,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
        'np': np,
    }
    return lambda x: eval(expr, {"__builtins__": {}}, {**safe_dict, 'x': x})


class GraphingCalculator(tk.Tk):
    """
    Main GUI application for the Graphing Calculator.
    
    Manages layout, inputs, safe function compilation, plotting, and evaluation.
    """

    def __init__(self):
        super().__init__()
        self.title('Graphing Calculator')
        self.geometry('1000x700')
        self.resizable(True, True)
        
        # Input variables
        self.xmin_var = tk.DoubleVar(value=-10.0)
        self.xmax_var = tk.DoubleVar(value=10.0)
        self.points_var = tk.IntVar(value=1000)
        self.x_eval_var = tk.DoubleVar(value=0.0)
        self.current_func = None
        
        # Frames
        top_frame = ttk.Frame(self)
        top_frame.pack(fill='x', padx=10, pady=5)
        middle_frame = ttk.Frame(self)
        middle_frame.pack(fill='both', expand=True)
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill='x', padx=10, pady=5)
        
        # Top frame: function input and parameters
        ttk.Label(top_frame, text='f(x) =').grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.func_entry = ttk.Entry(top_frame, width=60, font=('Arial', 12))
        self.func_entry.grid(row=0, column=1, columnspan=4, sticky='ew', padx=5)
        self.func_entry.insert(0, 'x**2')
        
        # Parameter inputs
        ttk.Label(top_frame, text='x_min:').grid(row=1, column=0, sticky='w', padx=(0, 5))
        ttk.Entry(top_frame, textvariable=self.xmin_var, width=8).grid(row=1, column=1, sticky='w', padx=2)
        ttk.Label(top_frame, text='x_max:').grid(row=1, column=2, sticky='w', padx=(10, 5))
        ttk.Entry(top_frame, textvariable=self.xmax_var, width=8).grid(row=1, column=3, sticky='w', padx=2)
        ttk.Label(top_frame, text='points:').grid(row=1, column=4, sticky='w', padx=(10, 5))
        ttk.Entry(top_frame, textvariable=self.points_var, width=8).grid(row=1, column=5, sticky='w', padx=2)
        
        ttk.Button(top_frame, text='Plot', command=self.plot_graph, width=10).grid(row=2, column=1, pady=5, padx=(0, 5))
        ttk.Button(top_frame, text='Clear', command=self.clear_plot, width=10).grid(row=2, column=2, pady=5, padx=(0, 10))
        
        top_frame.columnconfigure(1, weight=1)
        
        # Middle frame: plot canvas and toolbar
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, middle_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, middle_frame)
        self.toolbar.update()
        
        # Bottom frame: evaluator
        ttk.Label(bottom_frame, text='Evaluate at x:').grid(row=0, column=0, sticky='w', padx=(0, 5))
        ttk.Entry(bottom_frame, textvariable=self.x_eval_var, width=10).grid(row=0, column=1, padx=5, sticky='w')
        ttk.Button(bottom_frame, text='Eval', command=self.evaluate, width=8).grid(row=0, column=2, pady=5, padx=5)
        self.result_label = ttk.Label(bottom_frame, text='f(0.000) = ', font=('Arial', 12, 'bold'))
        self.result_label.grid(row=0, column=3, sticky='w', padx=10)
        
        # Initial plot
        self.plot_graph()

    def plot_graph(self) -> None:
        """Plots the function from inputs on the canvas."""
        try:
            expr = self.func_entry.get().strip()
            if not expr:
                raise ValueError('Enter a function')
            self.current_func = safe_compile(expr)
            xmin = self.xmin_var.get()
            xmax = self.xmax_var.get()
            if xmin >= xmax:
                raise ValueError('x_min must be less than x_max')
            points = int(self.points_var.get())
            if points < 2:
                raise ValueError('Number of points must be at least 2')
            x = np.linspace(xmin, xmax, points)
            y = self.current_func(x)
            self.ax.clear()
            self.ax.plot(x, y, 'b-', linewidth=2)
            self.ax.grid(True)
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_title('Graphing Calculator')
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror('Plot Error', str(e))

    def clear_plot(self) -> None:
        """Clears the plot axes and resets to empty state."""
        self.ax.clear()
        self.ax.set_title('Graphing Calculator')
        self.ax.grid(True)
        self.canvas.draw()

    def evaluate(self) -> None:
        """Evaluates the current function at x_eval_var and updates result label."""
        try:
            if self.current_func is None:
                raise ValueError('Plot a function first')
            xval = self.x_eval_var.get()
            yval = float(self.current_func(np.array([xval]))[0])
            self.result_label.config(text=f'f({xval:.3f}) = {yval:.6f}')
        except Exception as e:
            messagebox.showerror('Eval Error', str(e))


def main() -> None:
    """Application entry point."""
    app = GraphingCalculator()
    app.mainloop()


if __name__ == '__main__':
    main()