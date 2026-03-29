"""
CodeViz Analyzer - Python Code Visualizer and Analyzer

Setup/Usage:
1. pip install -r requirements.txt
2. python main.py
- Auto-loads sample1.py/sample2.py if present for demo.
- Load Python Files button for custom files.
- Test button triggers search+jump demo.
- Views: Code (syntax hl, line nums), Graph (interactive, hover/click), Search (semantic), Analysis (metrics table).
- Theme toggle, export PNG/SVG.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import os
from pygments import lexers
import pygments.token
import networkx as nx
from analyzer import CodeAnalyzer

class MainApp(ctk.CTk):
    """
    Main GUI application class handling UI setup, event bindings, view switching.
    Integrates CustomTkinter, Matplotlib canvas, Pygments highlighting, ttk Treeviews.
    """

    def __init__(self):
        super().__init__()
        self.title("CodeViz Analyzer")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Core attributes
        self.analyzer = CodeAnalyzer()
        self.current_file = None
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.hovered = None
        self.cached_pos = {}
        self.cached_G = None
        self.current_results = []

        # Theme colors
        self.dark_colors = {
            'default': '#f8f8f2',
            'keyword': '#ff79c6',
            'string': '#f1fa8c',
            'comment': '#6272a4',
            'function': '#50fa7b',
            'class_': '#8be9fd',
            'number': '#bd93f9',
        }
        self.light_colors = {
            'default': '#282828',
            'keyword': '#ff1493',
            'string': '#228b22',
            'comment': '#808080',
            'function': '#0066cc',
            'class_': '#1976d2',
            'number': '#b71c1c',
        }
        self.line_colors = {
            'dark': {'bg': '#282a36', 'fg': '#6272a4'},
            'light': {'bg': '#f8f8f2', 'fg': '#383a42'},
        }
        self.theme_var = ctk.BooleanVar(value=True)  # True = dark

        self.setup_ui()

        # Auto-demo load samples if present
        if os.path.exists('sample1.py'):
            self.analyzer.parse_file('sample1.py')
            self.file_tree.insert('', 'end', iid='sample1.py', text='sample1.py')
            print('DEMO-LOAD: sample1')
        if os.path.exists('sample2.py'):
            self.analyzer.parse_file('sample2.py')
            self.file_tree.insert('', 'end', iid='sample2.py', text='sample2.py')
            print('DEMO-LOAD: sample2')
        self.update_graph_view()
        print('DEMO-GRAPH updated')
        self.update_analysis()
        print('DEMO-ANALYSIS updated')

    def setup_ui(self):
        """Set up main UI: sidebar (ttk.Treeview), tabview, status."""
        # Left sidebar
        self.sidebar = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nswe", padx=5, pady=5)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(1, weight=1)

        load_btn = ctk.CTkButton(self.sidebar, text="Load Python Files", command=self.load_files, height=40)
        load_btn.pack(pady=20, padx=20)

        self.theme_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Dark Mode",
            variable=self.theme_var,
            command=self.toggle_theme,
        )
        self.theme_switch.pack(pady=10, padx=20)
        self.theme_switch.select()  # Start dark

        # Test button
        test_btn = ctk.CTkButton(self.sidebar, text="Test", command=self.test_demo)
        test_btn.pack(pady=10, padx=20)

        # File tree (ttk)
        self.file_tree = ttk.Treeview(self.sidebar, show="tree", height=15)
        self.file_tree.pack(fill="both", expand=True, pady=10, padx=20)
        self.file_tree.bind("<<TreeviewSelect>>", self.on_file_select)
        self.file_tree.bind("<Double-1>", self.on_file_select)

        # Central tabs
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

        self.code_tab = self.tabview.add("Code View")
        self.graph_tab = self.tabview.add("Graph View")
        self.search_tab = self.tabview.add("Search View")
        self.analysis_tab = self.tabview.add("Analysis View")

        self.setup_code_view()
        self.setup_graph_view()
        self.setup_search_view()
        self.setup_analysis_view()

        # Status
        self.status_label = ctk.CTkLabel(self, text="Ready - No files loaded")
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5, padx=10)

        # Grid config
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Initial theme
        self.toggle_theme()

    def setup_code_view(self):
        """Syntax-highlighted code with line numbers, scroll sync."""
        code_frame = ctk.CTkFrame(self.code_tab)
        code_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Lines
        lines_frame = ctk.CTkFrame(code_frame, width=60)
        lines_frame.pack(side="left", fill="y")
        self.lines_text = tk.Text(
            lines_frame, width=6, padx=3, takefocus=0, border=0, state="disabled",
            wrap="none", font=("Consolas", 10)
        )
        self.lines_text.pack(side="left", fill="y")

        # Code
        text_frame = ctk.CTkFrame(code_frame)
        text_frame.pack(side="left", fill="both", expand=True)
        self.code_text = tk.Text(
            text_frame, wrap="none", undo=True, padx=10, pady=10, border=0,
            state="disabled", font=("Consolas", 10)
        )
        self.code_text.pack(side="left", fill="both", expand=True)

        # Scrollbar
        yscroll = ctk.CTkScrollbar(text_frame, orientation="vertical", command=self.code_text.yview)
        yscroll.pack(side="right", fill="y")
        self.code_text.configure(yscrollcommand=yscroll.set)

        # Scroll sync
        self.code_text.bind("<MouseWheel>", self.on_mousewheel)
        self.lines_text.bind("<MouseWheel>", self.on_mousewheel_lines)

        # Tags
        for tag in list(self.dark_colors.keys()) + ['highlight']:
            self.code_text.tag_configure(tag, font=("Consolas", 10))

    def on_mousewheel(self, event):
        delta = int(-1 * (event.delta / 120))
        self.code_text.yview_scroll(delta, "units")
        self.lines_text.yview_scroll(delta, "units")
        return "break"

    def on_mousewheel_lines(self, event):
        self.on_mousewheel(event)

    def setup_graph_view(self):
        """Interactive Matplotlib call graph with toolbar, export."""
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.graph_tab)
        toolbar.update()

        # Events
        self.canvas.mpl_connect("button_press_event", self.on_node_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # Export
        export_frame = ctk.CTkFrame(self.graph_tab)
        export_frame.pack(fill="x", pady=10)
        ctk.CTkButton(export_frame, text="Export PNG", command=self.save_png).pack(side="left", padx=5)
        ctk.CTkButton(export_frame, text="Export SVG", command=self.save_svg).pack(side="left", padx=5)

    def setup_search_view(self):
        """Semantic search input, results Listbox, snippet label."""
        self.search_entry = ctk.CTkEntry(
            self.search_tab, placeholder_text="Enter search query..."
        )
        self.search_entry.pack(pady=20, padx=20, fill="x")
        self.search_entry.bind("<KeyRelease>", self.on_search)

        self.results_list = tk.Listbox(self.search_tab, height=15, font=("Consolas", 10))
        self.results_list.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        self.results_list.bind("<<ListboxSelect>>", self.on_result_select)

        self.snippet_label = ctk.CTkLabel(
            self.search_tab, text="", wraplength=1000, justify="left", font=("Consolas", 9)
        )
        self.snippet_label.pack(pady=10, padx=20, fill="x")

    def setup_analysis_view(self):
        """Metrics table with ttk.Treeview."""
        columns = ("File", "Num Elements", "Avg Comp", "Max Comp", "LOC", "Issues")
        self.metrics_tree = ttk.Treeview(
            self.analysis_tab, columns=columns, show="headings", height=20
        )
        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=120, anchor="w")
        self.metrics_tree.pack(fill="both", expand=True, padx=20, pady=20)

    def toggle_theme(self):
        """Toggle dark/light, update colors/tags/graph."""
        mode = "dark" if self.theme_var.get() else "light"
        ctk.set_appearance_mode(mode)

        colors = self.dark_colors if mode == "dark" else self.light_colors
        for tag, fg_color in colors.items():
            self.code_text.tag_configure(tag, foreground=fg_color)

        line_color = self.line_colors[mode]
        self.lines_text.configure(bg=line_color["bg"], fg=line_color["fg"])

        hl_bg = "#ffeb3b" if mode == "dark" else "#90caf9"
        self.code_text.tag_configure("highlight", background=hl_bg)

        # Graph bg
        bg_color = "#1e1e1e" if mode == "dark" else "white"
        ax_bg = "#2d2d2d" if mode == "dark" else "#f8f8f2"
        self.fig.patch.set_facecolor(bg_color)
        self.ax.set_facecolor(ax_bg)

        # Redraw graph if cached
        if hasattr(self, 'cached_G') and self.cached_G:
            self.update_graph_view()
            if hasattr(self, 'canvas'):
                self.canvas.draw_idle()

    def get_token_tag(self, token_type):
        """Map Pygments token to style tag."""
        if token_type in pygments.token.Keyword:
            return "keyword"
        elif token_type in pygments.token.Name.Function:
            return "function"
        elif token_type in pygments.token.Name.Class:
            return "class_"
        elif token_type in pygments.token.Literal.String:
            return "string"
        elif token_type in pygments.token.Comment:
            return "comment"
        elif token_type in pygments.token.Literal.Number:
            return "number"
        return "default"

    def highlight_code(self, code):
        """Pygments highlight + line numbers."""
        self.code_text.configure(state="normal")
        self.code_text.delete("1.0", tk.END)

        lines = code.splitlines()
        self.lines_text.configure(state="normal")
        self.lines_text.delete("1.0", tk.END)
        for i in range(1, len(lines) + 1):
            self.lines_text.insert(tk.END, f"{i}\n")
        self.lines_text.configure(state="disabled")

        lexer = lexers.get_lexer_by_name("python")
        for token_type, value in lexer.get_tokens(code):
            tag = self.get_token_tag(token_type)
            self.code_text.insert(tk.END, value, tag)

        self.code_text.configure(state="disabled")
        self.code_text.see("1.0")

    def load_files(self):
        """Load .py files, parse, update tree/graph/analysis."""
        files = filedialog.askopenfilenames(title="Select Python Files", filetypes=[("Python", "*.py")])
        if not files:
            return

        # Progress
        progress_frame = ctk.CTkFrame(self)
        progress_frame.pack(fill="x", pady=5)
        progress = ctk.CTkProgressBar(progress_frame, mode="indeterminate")
        progress.pack(pady=5, padx=10)
        progress.start()

        success = 0
        for fp in files[:50]:
            if self.analyzer.parse_file(fp):
                basename = os.path.basename(fp)
                self.file_tree.insert('', 'end', iid=fp, text=basename)
                success += 1

        progress.stop()
        progress_frame.destroy()

        total = len(self.analyzer.files)
        self.status_label.configure(text=f"Loaded {total} files ({success} success)")
        self.update_graph_view()
        self.update_analysis()

    def on_file_select(self, event):
        """Load/highlight selected file code."""
        selection = self.file_tree.selection()
        if not selection:
            return
        fp = selection[0]
        self.current_file = fp
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                code = f.read()
            self.highlight_code(code)
            self.status_label.configure(text=f"Viewing: {os.path.basename(fp)}")
        except Exception as e:
            self.status_label.configure(text=f"Error loading {os.path.basename(fp)}: {e}")

    def update_graph_view(self):
        """Draw merged call graph, cache pos/G, update status/print."""
        self.ax.clear()
        
        try:
            G = self.analyzer.get_merged_graph()
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        if len(G.nodes) == 0:
            self.ax.text(0.5, 0.5, "Load files to see call graph.", ha="center", va="center",
                         transform=self.ax.transAxes, fontsize=14)
            self.canvas.draw()
            return
        
        pos = nx.spring_layout(G, k=1, iterations=50)
        indeg = dict(G.in_degree())
        max_indeg = max(indeg.values() or [1])
        norm = Normalize(0, max_indeg)
        node_colors = [norm(indeg.get(n, 0)) for n in G.nodes]
        
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color=node_colors, cmap=plt.cm.Blues,
                               node_size=1200, alpha=0.85)
        nx.draw_networkx_edges(G, pos, ax=self.ax, node_size=1200, arrowsize=15, alpha=0.6)
        nx.draw_networkx_labels(G, pos, ax=self.ax, font_size=8, font_weight="bold")
        
        self.ax.set_title("Call Graph (nodes by in-degree)", fontsize=12)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Blues)
        plt.colorbar(sm, ax=self.ax, shrink=0.8)
        self.ax.axis("off")
        
        self.canvas.draw()
        self.cached_pos = pos
        self.cached_G = G
        print(f'GUI-GRAPH: {len(G.nodes)} nodes, {len(G.edges)} edges')
        self.status_label.configure(text=f'Graph: {len(G.nodes)} nodes, {len(G.edges)} edges')

    def on_hover(self, event):
        """Status update on node hover (dist < 0.05)."""
        if not hasattr(self, 'cached_pos') or not self.cached_pos or event.inaxes != self.ax:
            if self.hovered:
                self.hovered = None
                self.status_label.configure(text="")
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        dists = {n: ((self.cached_pos[n][0] - x)**2 + (self.cached_pos[n][1] - y)**2)**0.5
                 for n in self.cached_pos}
        closest = min(dists, key=dists.get)
        if dists[closest] < 0.05:
            if closest != self.hovered:
                self.hovered = closest
                info = self.analyzer.nodes.get(closest, {})
                typ = info.get('type', '?')
                comp = info.get('complexity', 'N/A')
                self.status_label.configure(text=f"Hovered: {closest} | {typ} | comp: {comp}")
        else:
            if self.hovered:
                self.hovered = None
                self.status_label.configure(text="")

    def on_node_click(self, event):
        """Jump to node on click (dist < 0.05)."""
        if (not hasattr(self, 'cached_pos') or not self.cached_pos or
            event.inaxes != self.ax or event.button != 1):
            return

        x, y = event.xdata, event.ydata
        dists = {n: ((self.cached_pos[n][0] - x)**2 + (self.cached_pos[n][1] - y)**2)**0.5
                 for n in self.cached_pos}
        closest = min(dists, key=dists.get)
        if dists[closest] < 0.05:
            self.jump_to_node(closest)

    def jump_to_node(self, node_id):
        """Jump to file/code location, highlight lines."""
        if node_id not in self.analyzer.nodes:
            print(f'GUI-JUMP: {node_id} not found')
            return
        info = self.analyzer.nodes[node_id]
        basename = node_id.split(':')[0]
        fps = [f for f in self.analyzer.files if os.path.basename(f) == basename]
        if not fps:
            print(f'GUI-JUMP: no fp for {basename}')
            return
        fp = fps[0]
        self.file_tree.selection_set(fp)
        self.on_file_select(None)
        lineno = info.get('lineno', 1)
        end = info.get('end_lineno', lineno)
        self.code_text.tag_add('highlight', f'{lineno}.0', f'{end + 1}.0')
        self.code_text.see(f'{lineno}.0')
        self.code_text.after(3000, lambda: self.code_text.tag_remove('highlight', '1.0', tk.END))
        print(f'GUI-JUMP: {node_id} lines {lineno}-{end}')

    def on_search(self, event=None):
        """Semantic search, populate listbox."""
        query = self.search_entry.get().strip()
        if not query:
            self.results_list.delete(0, tk.END)
            self.snippet_label.configure(text="")
            return
        results = self.analyzer.semantic_search(query)
        self.results_list.delete(0, tk.END)
        for r in results:
            self.results_list.insert(tk.END, f'{r[0]} ({r[1]:.2f})')
        self.current_results = results
        print(f'GUI-SEARCH: {len(results)} for "{query}"')

    def on_result_select(self, event):
        """Show snippet for selected result."""
        sel = self.results_list.curselection()
        if sel:
            node_id, score, snip = self.current_results[sel[0]]
            self.snippet_label.configure(text=f'{node_id} (score: {score:.3f})\n\n{snip}')

    def update_analysis(self):
        """Populate metrics Treeview, print summary."""
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        metrics = self.analyzer.get_metrics()
        file_count = len([k for k in metrics if k != "global"])
        for key, data in metrics.items():
            if key == 'global':
                values = ('GLOBAL', data['total_elements'], '-', data['max_complexity'], '-', data['num_cycles'])
            else:
                values = (key, data['num_elements'], f'{data["avg_complexity"]:.2f}',
                          data['max_complexity'], data['total_loc'], len(data['issues']))
            self.metrics_tree.insert('', 'end', values=values)
        print(f'GUI-ANALYSIS: {file_count} files + global')

    def test_demo(self):
        """Demo: search 'hello', jump to greet after 1s."""
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, 'hello')
        self.on_search()
        self.after(1000, lambda: self.jump_to_node('sample1.py:greet'))
        print('GUI-TEST: triggered search+jump')

    def save_png(self):
        """Export graph PNG."""
        fn = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if fn:
            self.fig.savefig(fn, dpi=300, bbox_inches='tight')
            print('GUI-EXPORT: PNG saved')

    def save_svg(self):
        """Export graph SVG."""
        fn = filedialog.asksaveasfilename(defaultextension='.svg', filetypes=[('SVG', '*.svg')])
        if fn:
            self.fig.savefig(fn, format='svg', bbox_inches='tight')
            print('GUI-EXPORT: SVG saved')

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()