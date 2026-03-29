import ast
import os
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import networkx as nx
from rapidfuzz import fuzz


class CodeAnalyzer:
    """
    Manages parsing, graph building, indexing, and analysis for loaded Python files.
    Key attributes:
    - graphs: dict[str, nx.DiGraph] - Per-file call graphs.
    - nodes: dict[str, dict] - Node info by prefixed ID (e.g., 'file.py:func').
    - files: list[str] - List of successfully loaded file paths.
    - index: defaultdict(list) - Search index 'all': list of (node_id, token_counter).
    - complexity_cache: dict[str, int] - Cached complexity per node_id.
    """

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Simple tokenizer: lowercase, extract word boundaries for bag-of-words.
        """
        return re.findall(r'\b\w+\b', text.lower())

    @staticmethod
    def cosine_sim(vec1: Counter, vec2: Counter) -> float:
        """
        Computes cosine similarity between two sparse Counter vectors (bag-of-words TF-IDF-like).
        """
        intersection = set(vec1.keys()) & set(vec2.keys())
        dot = sum(vec1[k] * vec2[k] for k in intersection)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    class CodeVisitor(ast.NodeVisitor):
        """
        AST visitor to extract definitions (functions/classes) and calls.
        Tracks current context (func/class) for proper naming and call attribution.
        """

        def __init__(self, lines: List[str]):
            super().__init__()
            self.lines = lines
            self.defs: List[Dict] = []
            self.calls: List[Tuple[str, str]] = []
            self.current_func: Optional[str] = None
            self.current_class: Optional[str] = None

        def _get_source(self, node: ast.AST) -> str:
            """
            Extracts source code snippet for a node using line numbers.
            Assumes Python 3.8+ with end_lineno support.
            """
            if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
                return ''
            start = node.lineno - 1
            end = node.end_lineno
            return '\n'.join(self.lines[start:end])

        def _compute_complexity(self, node: ast.AST) -> int:
            """
            Simple cyclomatic complexity: 1 + count of control flow nodes (If/For/While/Try+handlers)
            in the AST subtree. Recursive traversal.
            """
            count = 1
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While)):
                    count += 1
                elif isinstance(child, ast.Try):
                    count += 1 + len(child.handlers)
                count += self._compute_complexity(child)
            return count

        def _get_callee_name(self, func: ast.expr) -> Optional[str]:
            """
            Resolves callee name from Call.func expr.
            Handles Name, Attribute (self.attr -> class.attr, other.value.attr).
            Ignores complex expressions.
            """
            if isinstance(func, ast.Name):
                return func.id
            elif isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == 'self' and self.current_class:
                    return f"{self.current_class}.{func.attr}"
                val_name = self._get_callee_name(func.value)
                if val_name:
                    return f"{val_name}.{func.attr}"
                return func.attr
            return None

        def visit_ClassDef(self, node: ast.ClassDef):
            """
            Collect class definition info, update current_class context, recurse.
            """
            doc = ast.get_docstring(node) or ''
            info = {
                'name': node.name,
                'type': 'class',
                'docstring': doc,
                'params': 0,
                'lineno': node.lineno,
                'end_lineno': node.end_lineno,
                'source': self._get_source(node),
                'complexity': 0  # Classes do not have complexity
            }
            self.defs.append(info)
            prev_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = prev_class

        def visit_FunctionDef(self, node: ast.FunctionDef):
            """
            Collect function/method definition info (incl. complexity, params), update current_func, recurse.
            Prefix name with current_class if in class.
            """
            doc = ast.get_docstring(node) or ''
            if self.current_class:
                fname = f"{self.current_class}.{node.name}"
                ftype = 'method'
            else:
                fname = node.name
                ftype = 'function'
            info = {
                'name': fname,
                'type': ftype,
                'docstring': doc,
                'params': len(node.args.args),
                'lineno': node.lineno,
                'end_lineno': node.end_lineno,
                'source': self._get_source(node),
                'complexity': self._compute_complexity(node)
            }
            self.defs.append(info)
            prev_func = self.current_func
            self.current_func = fname
            self.generic_visit(node)
            self.current_func = prev_func

        def visit_Call(self, node: ast.Call):
            """
            Record calls only within current function context.
            Resolve callee name, ignore unresolved/complex.
            """
            if self.current_func:
                callee = self._get_callee_name(node.func)
                if callee:
                    self.calls.append((self.current_func, callee))
            self.generic_visit(node)

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.graphs: Dict[str, nx.DiGraph] = {}
        self.files: List[str] = []
        self.index = defaultdict(list)
        self.complexity_cache: Dict[str, int] = {}

    def parse_file(self, filepath: str) -> bool:
        """
        Parses a single Python file's AST, extracts defs/calls via visitor,
        builds subgraph, updates index/nodes/graphs. Returns True on success.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)
            lines = code.splitlines()
            visitor = self.CodeVisitor(lines)
            visitor.visit(tree)
            G = self._build_graph(filepath, visitor.defs, visitor.calls)
            self.graphs[filepath] = G
            self.files.append(filepath)
            return True
        except SyntaxError:
            # Graceful handling of syntax errors
            return False
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return False

    def _build_graph(self, filepath: str, defs: List[Dict], calls: List[Tuple[str, str]]) -> nx.DiGraph:
        """
        Constructs per-file Networkx DiGraph from collected defs/calls.
        Prefixes node_ids with basename:, adds to self.nodes/index/complexity_cache.
        Adds edges only if both caller/callee nodes exist (intra-file calls).
        """
        basename = os.path.basename(filepath)
        G = nx.DiGraph()
        # Add nodes and update caches/index
        for info in defs:
            node_id = f"{basename}:{info['name']}"
            G.add_node(node_id, **info)
            self.nodes[node_id] = info
            if 'complexity' in info:
                self.complexity_cache[node_id] = info['complexity']
            text = f"{info['name']} {info.get('docstring', '')} {info['source']}"
            vec = Counter(self.tokenize(text))
            self.index['all'].append((node_id, vec))
        # Add intra-file edges
        for caller, callee in calls:
            caller_id = f"{basename}:{caller}"
            callee_id = f"{basename}:{callee}"
            if caller_id in G.nodes and callee_id in G.nodes:
                G.add_edge(caller_id, callee_id)
        return G

    def get_merged_graph(self) -> nx.DiGraph:
        """
        Merges all per-file graphs into one (union of nodes/edges).
        Prefixes ensure no name clashes.
        """
        if self.graphs:
            return nx.compose_all(self.graphs.values())
        return nx.DiGraph()

    def semantic_search(self, query: str, threshold: float = 0.3) -> List[Tuple[str, float, str]]:
        """
        Semantic search using cosine sim on tokenized name/doc/source.
        Returns ranked list of (node_id, sim_score, snippet) above threshold.
        Fallback to fuzzy string matching on names if no cosine hits.
        """
        q_vec = Counter(self.tokenize(query))
        results = []
        for node_id, vec in self.index['all']:
            sim = self.cosine_sim(q_vec, vec)
            if sim > threshold:
                info = self.nodes.get(node_id, {})
                snippet = (info.get('source', '')[:200] + '...').strip()
                results.append((node_id, sim, snippet))
        results.sort(key=lambda x: x[1], reverse=True)
        if not results:
            # Fallback: rapidfuzz ratio on name
            fb_results = []
            for node_id, info in self.nodes.items():
                ratio = fuzz.ratio(query, info['name']) / 100.0
                if ratio > 0.7:
                    snippet = (info.get('source', '')[:200] + '...').strip()
                    fb_results.append((node_id, ratio, snippet))
            fb_results.sort(key=lambda x: x[1], reverse=True)
            return fb_results
        return results

    def compute_complexity(self, node: ast.FunctionDef) -> int:
        """
        Calculates simple cyclomatic complexity for a function AST node.
        Uses the same logic as in CodeVisitor.
        """
        visitor = self.CodeVisitor([])  # Dummy lines, not used in complexity
        return visitor._compute_complexity(node)

    def get_metrics(self) -> Dict[str, Dict]:
        """
        Aggregates analysis metrics per file and global.
        Per-file: num_elements, avg/max complexity, total LOC, issues (high complexity >10).
        Global: total elements, max complexity, number of simple cycles.
        """
        merged = self.get_merged_graph()
        from_file: Dict[str, List[str]] = defaultdict(list)
        total_loc: Dict[str, int] = defaultdict(int)
        # Group nodes by file
        for node_id in self.nodes:
            basename = node_id.split(':', 1)[0]
            from_file[basename].append(node_id)
        metrics: Dict[str, Dict] = {}
        for basename, node_ids in from_file.items():
            file_infos = [self.nodes[nid] for nid in node_ids]
            comps = [info.get('complexity', 0) for info in file_infos]
            num_el = len(file_infos)
            avg_comp = sum(comps) / num_el if num_el > 0 else 0
            max_comp = max(comps) if comps else 0
            loc_sum = sum(len(info['source'].splitlines()) for info in file_infos)
            issues = [info['name'] for info in file_infos if info.get('complexity', 0) > 10]
            metrics[basename] = {
                'num_elements': num_el,
                'avg_complexity': round(avg_comp, 2),
                'max_complexity': max_comp,
                'total_loc': loc_sum,
                'issues': issues
            }
        # Global metrics
        cycles = []
        try:
            cycles = list(nx.simple_cycles(merged))
        except:
            pass  # Graceful if graph issues
        metrics['global'] = {
            'total_elements': len(self.nodes),
            'max_complexity': max(self.complexity_cache.values()) if self.complexity_cache else 0,
            'num_cycles': len(cycles)
        }
        return metrics