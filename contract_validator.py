#!/usr/bin/env python3
"""
Contract validator for the data contract system.
Performs AST-based validation to ensure generated code matches contract specification.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


class ContractViolation(Exception):
    """Raised when code violates the data contract."""
    pass


class ContractValidator:
    """Validates Python code against contract specification."""
    
    def __init__(self, contract_path: Path, symbol_table_path: Optional[Path] = None):
        self.contract_path = contract_path
        self.contract = yaml.safe_load(contract_path.read_text())
        
        # Load symbol table
        if symbol_table_path and symbol_table_path.exists():
            self.symbol_table = json.loads(symbol_table_path.read_text())
        else:
            self.symbol_table = self._build_symbol_table()
        
        self.violations: List[str] = []
    
    def _build_symbol_table(self) -> Dict[str, Any]:
        """Build symbol table from contract specification."""
        symbols = {}
        
        # Add types
        for type_name, type_spec in self.contract.get('types', {}).items():
            symbols[type_name] = {
                'kind': 'type',
                'fields': type_spec.get('fields', {}),
                'spec': type_spec
            }
        
        # Add classes
        for class_name, class_spec in self.contract.get('classes', {}).items():
            methods = {}
            for method_name, method_spec in class_spec.get('methods', {}).items():
                methods[method_name] = {
                    'args': method_spec.get('args', {}),
                    'returns': method_spec.get('returns', 'None')
                }
            
            symbols[class_name] = {
                'kind': 'class',
                'init': class_spec.get('init', {}),
                'attrs': class_spec.get('attrs', {}),
                'methods': methods,
                'spec': class_spec
            }
        
        # Add functions
        for func_name, func_spec in self.contract.get('functions', {}).items():
            symbols[func_name] = {
                'kind': 'function',
                'args': func_spec.get('args', {}),
                'returns': func_spec.get('returns', 'None'),
                'spec': func_spec
            }
        
        # Add constants
        for const_name, const_value in self.contract.get('constants', {}).items():
            symbols[const_name] = {
                'kind': 'constant',
                'value': const_value
            }
        
        return symbols
    
    def validate_code(self, code: str, filename: str = "<string>") -> bool:
        """Validate Python code against the contract."""
        self.violations.clear()
        
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            self.violations.append(f"Syntax error: {e}")
            return False
        
        # Walk the AST and validate
        validator = ASTValidator(self.symbol_table, filename)
        validator.visit(tree)
        
        self.violations.extend(validator.violations)
        return len(self.violations) == 0
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate a Python file against the contract."""
        try:
            code = file_path.read_text()
            return self.validate_code(code, str(file_path))
        except Exception as e:
            self.violations.append(f"Error reading file {file_path}: {e}")
            return False
    
    def get_violations(self) -> List[str]:
        """Get list of contract violations."""
        return self.violations.copy()
    
    def print_violations(self):
        """Print contract violations to stderr."""
        if self.violations:
            print("Contract violations found:", file=sys.stderr)
            for violation in self.violations:
                print(f"  • {violation}", file=sys.stderr)
        else:
            print("No contract violations found.")


class ASTValidator(ast.NodeVisitor):
    """AST visitor that validates code against contract symbols."""
    
    def __init__(self, symbol_table: Dict[str, Any], filename: str):
        self.symbol_table = symbol_table
        self.filename = filename
        self.violations: List[str] = []
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.local_vars: Set[str] = set()
        self.imported_symbols: Set[str] = set()
    
    def _add_violation(self, node: ast.AST, message: str):
        """Add a contract violation with location info."""
        lineno = getattr(node, 'lineno', '?')
        col_offset = getattr(node, 'col_offset', '?')
        location = f"{self.filename}:{lineno}:{col_offset}"
        self.violations.append(f"{location}: {message}")
    
    def visit_Import(self, node: ast.Import):
        """Track imported symbols."""
        for alias in node.names:
            self.imported_symbols.add(alias.name)
            if alias.asname:
                self.imported_symbols.add(alias.asname)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track imported symbols from modules."""
        for alias in node.names:
            self.imported_symbols.add(alias.name)
            if alias.asname:
                self.imported_symbols.add(alias.asname)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Validate class definition against contract."""
        old_class = self.current_class
        self.current_class = node.name
        
        # Check if class is defined in contract
        if node.name in self.symbol_table:
            class_spec = self.symbol_table[node.name]
            # Allow both 'class' and 'type' (dataclasses) kinds
            if class_spec['kind'] not in ['class', 'type']:
                self._add_violation(node, f"'{node.name}' should be a class but is defined as {class_spec['kind']}")
        else:
            self._add_violation(node, f"Class '{node.name}' not found in contract")
        
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Validate function definition against contract."""
        old_function = self.current_function
        self.current_function = node.name
        
        # Clear local variables for new function scope
        old_locals = self.local_vars.copy()
        self.local_vars.clear()
        
        # Add function parameters to local variables
        for arg in node.args.args:
            self.local_vars.add(arg.arg)
        
        # Validate function signature
        if self.current_class:
            # Method validation
            if self.current_class in self.symbol_table:
                class_spec = self.symbol_table[self.current_class]
                if node.name == '__init__':
                    # Validate __init__ method
                    expected_params = class_spec.get('init', {})
                    self._validate_function_signature(node, expected_params, 'None')
                elif node.name in class_spec.get('methods', {}):
                    # Validate regular method
                    method_spec = class_spec['methods'][node.name]
                    self._validate_function_signature(node, method_spec['args'], method_spec['returns'])
                else:
                    self._add_violation(node, f"Method '{node.name}' not defined in contract for class '{self.current_class}'")
        else:
            # Standalone function validation
            if node.name in self.symbol_table:
                func_spec = self.symbol_table[node.name]
                if func_spec['kind'] == 'function':
                    self._validate_function_signature(node, func_spec['args'], func_spec['returns'])
                else:
                    self._add_violation(node, f"'{node.name}' should be a function but is defined as {func_spec['kind']}")
            else:
                # Allow functions not in contract (internal functions)
                pass
        
        self.generic_visit(node)
        
        # Restore previous state
        self.current_function = old_function
        self.local_vars = old_locals
    
    def _validate_function_signature(self, node: ast.FunctionDef, expected_args: Dict[str, str], expected_return: str):
        """Validate function signature against expected specification."""
        # Get actual arguments (excluding 'self' for methods)
        actual_args = node.args.args
        if self.current_class and actual_args and actual_args[0].arg == 'self':
            actual_args = actual_args[1:]
        
        # Check argument count
        if len(actual_args) != len(expected_args):
            self._add_violation(node, 
                f"Function '{node.name}' expects {len(expected_args)} arguments, got {len(actual_args)}")
        
        # Check argument names (order matters)
        for i, (expected_name, expected_type) in enumerate(expected_args.items()):
            if i < len(actual_args):
                actual_arg = actual_args[i]
                if actual_arg.arg != expected_name:
                    self._add_violation(node, 
                        f"Argument {i+1} of '{node.name}' should be '{expected_name}', got '{actual_arg.arg}'")
                
                # TODO: Validate type annotations if present
    
    def visit_Name(self, node: ast.Name):
        """Validate name references."""
        name = node.id
        
        # Skip validation for certain contexts
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            if isinstance(node.ctx, ast.Store):
                self.local_vars.add(name)
            self.generic_visit(node)
            return
        
        # Check if name is valid
        builtin_types = {'str', 'int', 'float', 'bool', 'dict', 'list', 'tuple', 'set', 'object', 'type', 'None'}
        
        if (name not in self.symbol_table and 
            name not in self.local_vars and 
            name not in self.imported_symbols and
            name not in {'self', 'cls'} and
            name not in builtin_types and
            not name.startswith('_')):  # Allow private variables
            
            # Check if it's a built-in
            if name not in dir(__builtins__):
                self._add_violation(node, f"Undefined symbol '{name}' not found in contract")
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Validate attribute access."""
        # Get the base object type if possible
        if isinstance(node.value, ast.Name):
            base_name = node.value.id
            attr_name = node.attr
            
            # Check if base is a known class/type
            if base_name in self.symbol_table:
                symbol_spec = self.symbol_table[base_name]
                
                if symbol_spec['kind'] == 'class':
                    # Check if attribute exists in class
                    attrs = symbol_spec.get('attrs', {})
                    methods = symbol_spec.get('methods', {})
                    
                    if (attr_name not in attrs and 
                        attr_name not in methods and
                        not attr_name.startswith('_')):  # Allow private attributes
                        self._add_violation(node, 
                            f"Attribute '{attr_name}' not defined for class '{base_name}'")
                
                elif symbol_spec['kind'] == 'type':
                    # Check if field exists in dataclass
                    fields = symbol_spec.get('fields', {})
                    if (attr_name not in fields and
                        not attr_name.startswith('_')):
                        self._add_violation(node,
                            f"Field '{attr_name}' not defined for type '{base_name}'")
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Validate function/method calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Validate function call
            if func_name in self.symbol_table:
                func_spec = self.symbol_table[func_name]
                if func_spec['kind'] == 'function':
                    expected_args = func_spec.get('args', {})
                    self._validate_call_arguments(node, func_name, expected_args)
                elif func_spec['kind'] == 'class':
                    # Constructor call
                    expected_args = func_spec.get('init', {})
                    self._validate_call_arguments(node, f"{func_name}.__init__", expected_args)
        
        elif isinstance(node.func, ast.Attribute):
            # Method call
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr
                
                if obj_name in self.symbol_table:
                    obj_spec = self.symbol_table[obj_name]
                    if obj_spec['kind'] == 'class':
                        methods = obj_spec.get('methods', {})
                        if method_name in methods:
                            expected_args = methods[method_name].get('args', {})
                            self._validate_call_arguments(node, f"{obj_name}.{method_name}", expected_args)
        
        self.generic_visit(node)
    
    def _validate_call_arguments(self, node: ast.Call, func_name: str, expected_args: Dict[str, str]):
        """Validate function call arguments."""
        # Count positional arguments
        num_positional = len(node.args)
        
        # Count keyword arguments
        keyword_names = {kw.arg for kw in node.keywords if kw.arg}
        
        # Check if all required arguments are provided
        total_provided = num_positional + len(keyword_names)
        
        # For now, just check argument count (more sophisticated validation could be added)
        if total_provided > len(expected_args):
            self._add_violation(node, 
                f"Call to '{func_name}' has too many arguments: expected {len(expected_args)}, got {total_provided}")


def main():
    """Main entry point for contract validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate code against contract specification")
    parser.add_argument("--contract", default="contract.yml", help="Path to contract specification")
    parser.add_argument("--symbols", help="Path to symbol table JSON file")
    parser.add_argument("files", nargs="+", help="Python files to validate")
    
    args = parser.parse_args()
    
    contract_path = Path(args.contract)
    symbol_table_path = Path(args.symbols) if args.symbols else None
    
    if not contract_path.exists():
        print(f"Error: Contract specification not found: {contract_path}", file=sys.stderr)
        return 1
    
    validator = ContractValidator(contract_path, symbol_table_path)
    
    all_valid = True
    for file_path in args.files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            all_valid = False
            continue
        
        print(f"Validating {file_path}...")
        if not validator.validate_file(file_path):
            all_valid = False
            validator.print_violations()
    
    if all_valid:
        print("✓ All files pass contract validation")
        return 0
    else:
        print("✗ Contract validation failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())