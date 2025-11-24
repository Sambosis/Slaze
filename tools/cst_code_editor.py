from __future__ import annotations

from enum import Enum
from typing import Literal, List, Optional, Tuple, Dict, Union
from pathlib import Path
import os
import logging
import difflib
import hashlib
import textwrap

# type: ignore[override]
from .base import ToolResult, BaseAnthropicTool
from config import get_constant

try:
    import libcst as cst
    from libcst import matchers as m
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False

from rich import print as rr
logger = logging.getLogger(__name__)

# ----------------------------
# Utility
# ----------------------------

def _within_repo(repo: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(repo.resolve())
        return True
    except Exception:
        return False

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _diff(a: str, b: str) -> str:
    return "".join(difflib.unified_diff(
        a.splitlines(keepends=True),
        b.splitlines(keepends=True),
        fromfile="before", tofile="after"
    ))

# ----------------------------
# CST Visitors / Transformers
# ----------------------------

class SymbolCollector(cst.CSTVisitor):
    def __init__(self):
        self.symbols = []
        self.stack = []

    def visit_ClassDef(self, node: cst.ClassDef):
        self.stack.append(node.name.value)
        self._add_symbol("class", node)

    def leave_ClassDef(self, node: cst.ClassDef):
        self.stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef):
        name = node.name.value
        full_name = ".".join(self.stack + [name])
        kind = "method" if self.stack else "function"
        self._add_symbol(kind, node, full_name)
        self.stack.append(name)

    def leave_FunctionDef(self, node: cst.FunctionDef):
        self.stack.pop()

    def visit_Assign(self, node: cst.Assign):
        # Heuristic: only top-level or class-level assignments
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                name = target.target.value
                full_name = ".".join(self.stack + [name])
                self._add_symbol("variable", node, full_name)

    def visit_AnnAssign(self, node: cst.AnnAssign):
        if isinstance(node.target, cst.Name):
            name = node.target.value
            full_name = ".".join(self.stack + [name])
            self._add_symbol("variable", node, full_name)

    def _add_symbol(self, kind: str, node: cst.CSTNode, name: str = None):
        if name is None:
            if hasattr(node, 'name') and hasattr(node.name, 'value'):
                name = ".".join(self.stack) # Stack already includes current node name for ClassDef
            else:
                return # Should not happen for Class/Func
        
        # CST doesn't give line numbers by default unless we use MetadataWrapper, 
        # but for simple listing we might skip lines or use a wrapper if needed.
        # For now, we'll just list names.
        self.symbols.append({"kind": kind, "symbol": name})

class SymbolFinder(cst.CSTVisitor):
    def __init__(self, target_symbol: str):
        self.target_symbol = target_symbol
        self.found_node = None
        self.stack = []

    def visit_ClassDef(self, node: cst.ClassDef):
        self.stack.append(node.name.value)
        if ".".join(self.stack) == self.target_symbol:
            self.found_node = node
            # Don't stop visiting, we might need to traverse children if we were looking for a child
            # But actually if we found it, we can stop? No, because we might be looking for a nested symbol with same prefix?
            # Actually, exact match means we found it.
            return False # Stop traversing children of this node? No, what if target is inside?
            # Wait, if ".".join(self.stack) == target, then THIS node is the target.
        
    def leave_ClassDef(self, node: cst.ClassDef):
        self.stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef):
        self.stack.append(node.name.value)
        if ".".join(self.stack) == self.target_symbol:
            self.found_node = node
        
    def leave_FunctionDef(self, node: cst.FunctionDef):
        self.stack.pop()
    
    def visit_Assign(self, node: cst.Assign):
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                if ".".join(self.stack + [target.target.value]) == self.target_symbol:
                    self.found_node = node

    def visit_AnnAssign(self, node: cst.AnnAssign):
        if isinstance(node.target, cst.Name):
            if ".".join(self.stack + [node.target.value]) == self.target_symbol:
                self.found_node = node

class EditorTransformer(cst.CSTTransformer):
    def __init__(self, command: str, target_symbol: str, text: str, keep_docstring: bool = True):
        self.command = command
        self.target_symbol = target_symbol
        self.text = text
        self.keep_docstring = keep_docstring
        self.stack = []
        self.modified = False

    def _is_target(self, name: str) -> bool:
        return ".".join(self.stack + [name]) == self.target_symbol


    
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> Union[cst.FunctionDef, cst.FlattenSentinel[cst.FunctionDef], cst.RemovalSentinel]:
        current_symbol = ".".join(self.stack)
        res = updated_node
        if current_symbol == self.target_symbol:
            res = self._apply_edit(original_node, updated_node)
        self.stack.pop()
        return res

    def visit_FunctionDef(self, node: cst.FunctionDef):
        self.stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> Union[cst.ClassDef, cst.FlattenSentinel[cst.ClassDef], cst.RemovalSentinel]:
        current_symbol = ".".join(self.stack)
        res = updated_node
        if current_symbol == self.target_symbol:
            res = self._apply_edit(original_node, updated_node)
        self.stack.pop()
        return res

    def visit_ClassDef(self, node: cst.ClassDef):
        self.stack.append(node.name.value)

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> Union[cst.Assign, cst.FlattenSentinel[cst.Assign], cst.RemovalSentinel]:
        # Check if any target matches
        for target in original_node.targets:
            if isinstance(target.target, cst.Name):
                if self._is_target(target.target.value):
                    return self._apply_edit(original_node, updated_node)
        return updated_node

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> Union[cst.AnnAssign, cst.FlattenSentinel[cst.AnnAssign], cst.RemovalSentinel]:
        if isinstance(original_node.target, cst.Name):
            if self._is_target(original_node.target.value):
                return self._apply_edit(original_node, updated_node)
        return updated_node

    def _apply_edit(self, original_node, updated_node):
        self.modified = True
        
        if self.command == "replace_whole":
            try:
                new_node = cst.parse_statement(self.text)
                return new_node
            except Exception as e:
                raise ValueError(f"Failed to parse replacement text: {e}")

        elif self.command == "replace_body":
            # Only for Class/Func
            if not isinstance(updated_node, (cst.ClassDef, cst.FunctionDef)):
                raise ValueError("replace_body only supports classes and functions")
            
            try:
                # Dedent input text to ensure it's a valid module body
                dedented_text = textwrap.dedent(self.text)
                new_module = cst.parse_module(dedented_text)
                new_body = new_module.body
            except Exception as e:
                raise ValueError(f"Failed to parse body text: {e}")

            # Handle docstring
            if self.keep_docstring:
                doc = updated_node.get_docstring()
                if doc:
                    old_body = updated_node.body.body
                    if old_body and m.matches(old_body[0], m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())])):
                        doc_node = old_body[0]
                        if not (new_body and m.matches(new_body[0], m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())]))):
                            new_body = [doc_node] + list(new_body)
            
            return updated_node.with_changes(body=cst.IndentedBlock(body=new_body))

        elif self.command == "insert_before":
            try:
                new_module = cst.parse_module(textwrap.dedent(self.text))
                new_nodes = new_module.body
                return cst.FlattenSentinel(list(new_nodes) + [updated_node])
            except Exception as e:
                raise ValueError(f"Failed to parse insert text: {e}")

        elif self.command == "insert_after":
             try:
                new_module = cst.parse_module(textwrap.dedent(self.text))
                new_nodes = new_module.body
                return cst.FlattenSentinel([updated_node] + list(new_nodes))
             except Exception as e:
                raise ValueError(f"Failed to parse insert text: {e}")

        elif self.command == "add_decorator":
            if not isinstance(updated_node, (cst.ClassDef, cst.FunctionDef)):
                raise ValueError("add_decorator only supports classes and functions")
            
            # Parse decorator
            # Text should be "@decorator" or "decorator"
            dec_text = self.text.strip()
            if not dec_text.startswith("@"):
                dec_text = "@" + dec_text
            
            try:
                # Parse as a module to get the decorator node?
                # cst.parse_module("@dec\ndef f(): pass")
                dummy = f"{dec_text}\ndef dummy(): pass"
                dummy_module = cst.parse_module(dummy)
                new_dec = dummy_module.body[0].decorators[0]
                
                # Add to existing decorators
                return updated_node.with_changes(decorators=list(updated_node.decorators) + [new_dec])
            except Exception as e:
                raise ValueError(f"Failed to parse decorator: {e}")

        elif self.command == "remove_decorator":
            if not isinstance(updated_node, (cst.ClassDef, cst.FunctionDef)):
                raise ValueError("remove_decorator only supports classes and functions")
            
            target_dec = self.text.strip()
            if target_dec.startswith("@"):
                target_dec = target_dec[1:]
            
            new_decs = []
            for dec in updated_node.decorators:
                # Extract name from decorator
                # Decorator.decorator is the expression
                dec_code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=dec.decorator)])]).code.strip()
                # This is a bit hacky.
                # Better: use cst.ensure_type(dec.decorator, cst.Name) or cst.Call
                # Let's just compare the source code of the decorator expression
                # But we need to be careful about whitespace?
                # Let's try to match the name.
                
                # Simple approach: check if target_dec is in the string representation
                # Or parse target_dec and compare?
                
                # Let's just use string matching on the decorator expression code
                # We can get the code of the expression
                # But we don't have a good way to get code of a node without Module wrapper
                # Re-using the wrapper trick
                dec_expr_code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=dec.decorator)])]).code.strip()
                
                if dec_expr_code == target_dec or dec_expr_code.startswith(target_dec + "("):
                     self.modified = True
                     continue # Remove
                new_decs.append(dec)
            
            if not self.modified:
                 # Try looser match?
                 pass
            
            return updated_node.with_changes(decorators=new_decs)

        elif self.command == "wrap_body":
            if not isinstance(updated_node, (cst.ClassDef, cst.FunctionDef)):
                raise ValueError("wrap_body only supports classes and functions")
            
            # Parse wrapper
            try:
                wrapper_module = cst.parse_module(textwrap.dedent(self.text))
                wrapper_body = wrapper_module.body
            except Exception as e:
                raise ValueError(f"Failed to parse wrapper text: {e}")
            
            # Find the hole (pass)
            # We look for a 'pass' statement in the wrapper
            # Recursively? Or just top level of wrapper?
            # Usually wrapper is `try: ...` so hole is inside `try` body.
            
            class HoleFinder(cst.CSTTransformer):
                def __init__(self, content_nodes):
                    self.content_nodes = content_nodes
                    self.filled = False
                
                def leave_Pass(self, original_node, updated_node):
                    if not self.filled:
                        self.filled = True
                        # Replace 'pass' with content_nodes
                        # But 'pass' is a SmallStatement, content_nodes are Statements (SimpleStatementLine usually)
                        # We can't replace SmallStatement with list of Statements directly if it's inside SimpleStatementLine
                        # We need to replace the SimpleStatementLine containing pass?
                        return cst.RemovalSentinel.REMOVE # Placeholder, handled by parent?
                    return updated_node
                
                def leave_SimpleStatementLine(self, original_node, updated_node):
                    # Check if this line contained the pass we removed
                    # If leave_Pass returned REMOVE, updated_node.body might be empty?
                    # Actually, if we want to inject a block, we need to be at the IndentedBlock level.
                    return updated_node

            # Better approach: Manual traversal to find the IndentedBlock containing 'pass'
            # and replace 'pass' with our nodes.
            
            def fill_hole(nodes, content):
                new_nodes = []
                filled = False
                for node in nodes:
                    if isinstance(node, cst.SimpleStatementLine):
                        # Check if it is just 'pass'
                        if len(node.body) == 1 and isinstance(node.body[0], cst.Pass):
                            # Found the hole!
                            new_nodes.extend(content)
                            filled = True
                            continue
                    
                    # Recurse into control structures
                    if isinstance(node, cst.Try):
                        new_body, f = fill_hole(node.body.body, content)
                        if f: filled = True
                        node = node.with_changes(body=node.body.with_changes(body=new_body))
                        # Also check handlers, else, finally?
                    elif isinstance(node, cst.With):
                        new_body, f = fill_hole(node.body.body, content)
                        if f: filled = True
                        node = node.with_changes(body=node.body.with_changes(body=new_body))
                    elif isinstance(node, cst.If):
                        new_body, f = fill_hole(node.body.body, content)
                        if f: filled = True
                        node = node.with_changes(body=node.body.with_changes(body=new_body))
                    # ... add others as needed
                    
                    new_nodes.append(node)
                return new_nodes, filled

            # Extract original body
            original_body = updated_node.body.body
            
            # Fill hole
            new_wrapper_body, filled = fill_hole(wrapper_body, original_body)
            
            if not filled:
                raise ValueError("Could not find 'pass' statement in wrapper to replace")
            
            return updated_node.with_changes(body=cst.IndentedBlock(body=new_wrapper_body))

        elif self.command == "rename":
            if isinstance(updated_node, (cst.ClassDef, cst.FunctionDef)):
                return updated_node.with_changes(name=cst.Name(value=self.text))
            elif isinstance(updated_node, cst.Assign):
                # Rename targets?
                # This is ambiguous if there are multiple targets.
                # We only support renaming if we can identify which target.
                # But EditorTransformer finds the node by matching ANY target.
                # We should probably rename ALL targets that match?
                new_targets = []
                for target in updated_node.targets:
                    if isinstance(target.target, cst.Name) and target.target.value == self.target_symbol.split(".")[-1]:
                        new_targets.append(target.with_changes(target=cst.Name(value=self.text)))
                    else:
                        new_targets.append(target)
                return updated_node.with_changes(targets=new_targets)
            elif isinstance(updated_node, cst.AnnAssign):
                 if isinstance(updated_node.target, cst.Name) and updated_node.target.value == self.target_symbol.split(".")[-1]:
                     return updated_node.with_changes(target=cst.Name(value=self.text))
            
            return updated_node
        
        elif self.command == "delete_symbol":
            return cst.RemovalSentinel.REMOVE

        return updated_node

class ImportTransformer(cst.CSTTransformer):
    def __init__(self, command: str, text: str):
        self.command = command
        self.text = text
        self.modified = False
        self.imports_seen = False
        self.last_import_index = -1
        self.body_nodes = []

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if self.command == "add_import":
            # Parse new import
            try:
                new_imp = cst.parse_statement(self.text)
            except Exception as e:
                raise ValueError(f"Invalid import text: {e}")

            # Find where to insert
            # We want to insert after the last import
            # Or at top if no imports
            
            body = list(updated_node.body)
            last_imp_idx = -1
            for i, node in enumerate(body):
                if isinstance(node, (cst.Import, cst.ImportFrom)):
                    last_imp_idx = i
            
            if last_imp_idx != -1:
                body.insert(last_imp_idx + 1, new_imp)
            else:
                # Insert at top, but after docstring/comments if possible
                # Heuristic: if first node is docstring, insert after
                idx = 0
                if body and m.matches(body[0], m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())])):
                    idx = 1
                body.insert(idx, new_imp)
            
            self.modified = True
            return updated_node.with_changes(body=body)

        elif self.command == "remove_import":
             # We need to match the import text to a node
             # This is hard to do exactly by text without parsing it.
             # Let's parse the text to get a structure, then match.
             try:
                 target_imp = cst.parse_statement(self.text)
             except:
                 raise ValueError(f"Invalid import text: {self.text}")

             new_body = []
             for node in updated_node.body:
                 # Compare node structure?
                 # cst.ensure_type(node, type(target_imp))
                 # This is tricky. Let's just compare the code strings for now?
                 # Or use deep_equals?
                 if node.deep_equals(target_imp):
                     self.modified = True
                     continue # Skip (remove)
                 new_body.append(node)
             
             if not self.modified:
                 # Try looser matching? E.g. if target is "import os", match "import os"
                 pass

             return updated_node.with_changes(body=new_body)

        return updated_node

class RefactorTransformer(cst.CSTTransformer):
    """
    Renames a symbol and all its references throughout the file.
    More comprehensive than 'rename' which only changes the definition.
    """
    def __init__(self, target_symbol: str, new_name: str):
        self.target_symbol = target_symbol
        self.new_name = new_name
        self.old_name = target_symbol.split(".")[-1]  # Last part of the symbol
        self.stack = []
        self.modified = False
        self.is_class = False
        self.is_method = False
        
    def visit_ClassDef(self, node: cst.ClassDef):
        current_symbol = ".".join(self.stack + [node.name.value])
        if current_symbol == self.target_symbol:
            self.is_class = True
        self.stack.append(node.name.value)
        
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef):
        current_symbol = ".".join(self.stack)
        if current_symbol == self.target_symbol:
            # Rename the class definition itself
            self.modified = True
            updated_node = updated_node.with_changes(name=cst.Name(value=self.new_name))
        self.stack.pop()
        return updated_node
    
    def visit_FunctionDef(self, node: cst.FunctionDef):
        current_symbol = ".".join(self.stack + [node.name.value])
        if current_symbol == self.target_symbol:
            self.is_method = len(self.stack) > 0  # Inside a class
        self.stack.append(node.name.value)
        
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef):
        current_symbol = ".".join(self.stack)
        if current_symbol == self.target_symbol:
            # Rename the function/method definition itself
            self.modified = True
            updated_node = updated_node.with_changes(name=cst.Name(value=self.new_name))
        self.stack.pop()
        return updated_node
    
    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name):
        # Rename references to the symbol
        if original_node.value == self.old_name:
            # Basic heuristic: rename all instances of the old name to new name
            # This is simplistic but works for many common cases
            # A more sophisticated approach would use scope analysis
            self.modified = True
            return updated_node.with_changes(value=self.new_name)
        return updated_node
    
    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute):
        # Handle method calls like self.method_name() or obj.method_name()
        if isinstance(updated_node.attr, cst.Name) and updated_node.attr.value == self.old_name:
            # Check if this matches our target (for methods)
            # This is a simplified check - ideally we'd track the type of 'self'
            self.modified = True
            return updated_node.with_changes(attr=cst.Name(value=self.new_name))
        return updated_node
    
    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign):
        # Handle variable assignments
        for target in original_node.targets:
            if isinstance(target.target, cst.Name):
                full_name = ".".join(self.stack + [target.target.value])
                if full_name == self.target_symbol:
                    # Rename the variable assignment
                    new_targets = []
                    for t in updated_node.targets:
                        if isinstance(t.target, cst.Name) and t.target.value == self.old_name:
                            new_targets.append(t.with_changes(target=cst.Name(value=self.new_name)))
                            self.modified = True
                        else:
                            new_targets.append(t)
                    return updated_node.with_changes(targets=new_targets)
        return updated_node
    
    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign):
        # Handle annotated variable assignments
        if isinstance(original_node.target, cst.Name):
            full_name = ".".join(self.stack + [original_node.target.value])
            if full_name == self.target_symbol:
                self.modified = True
                return updated_node.with_changes(target=cst.Name(value=self.new_name))
        return updated_node

# ----------------------------
# Tool Definition
# ----------------------------

class CSTEditorCommand(str, Enum):
    LIST = "list_symbols"
    SHOW = "show_symbol"
    REPLACE_WHOLE = "replace_whole"
    REPLACE_BODY = "replace_body"
    INSERT_AFTER = "insert_after"
    INSERT_BEFORE = "insert_before"
    DELETE = "delete_symbol"
    ADD_IMPORT = "add_import"
    REMOVE_IMPORT = "remove_import"
    ADD_DECORATOR = "add_decorator"
    REMOVE_DECORATOR = "remove_decorator"
    WRAP_BODY = "wrap_body"
    RENAME = "rename"
    REFACTOR = "refactor"

class CSTCodeEditorTool(BaseAnthropicTool):
    """
    Robust Python code editor using LibCST.
    Preserves formatting and comments.
    """
    name: Literal["cst_code_editor"] = "cst_code_editor"
    api_type: Literal["custom"] = "custom"
    description: str = (
        "Edit Python source files using LibCST for format-preserving AST modifications. "
        "Commands: list_symbols, show_symbol, replace_whole, replace_body, insert_after, insert_before, delete_symbol, add_import, remove_import, add_decorator, remove_decorator, wrap_body, rename. "
        "Superior to ast_code_editor for complex edits."
    )

    def __init__(self, display=None):
        super().__init__(display=display)
        self.display = display

    def _emit_console(self, cwd: str, action: str, detail: str = "", path: Optional[str] = None, symbol: Optional[str] = None) -> None:
        if self.display is None:
            return

        command_line = f"$ {action}"
        if path:
            command_line += f" path='{path}'"
        if symbol:
            command_line += f" symbol='{symbol}'"

        block = ["```console"]
        block.append(f"$ cd {cwd}")
        block.append(command_line)
        if detail:
            block.extend(detail.rstrip().splitlines())
        block.append("```")
        self.display.add_message("assistant", "\n".join(block))

    def to_params(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Edit a Python source file using LibCST in a format-preserving way. "
                    "Use this tool for small, surgical edits (renaming a symbol, replacing a body, "
                    "adding imports or decorators), not for rewriting whole files from scratch. "
                    "For a new file, create it by other means first, then use this tool to refine it."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "REQUIRED. Path to the Python file to edit, relative to the repository root.\n"
                                "This parameter is MANDATORY for all commands.\n"
                                "Example: 'src/main.py' or 'tools/cst_code_editor.py'"
                            ),
                        },
                        "command": {
                            "type": "string",
                            "enum": [c.value for c in CSTEditorCommand],
                            "description": (
                                "High-level editing action to perform on the file.\n\n"
                                "Valid commands and their expected arguments:\n"
                                "- list_symbols:\n"
                                "    List all top-level and class-level symbols in the file.\n"
                                "    Required: path\n"
                                "    Ignores: symbol, text, keep_docstring, dry_run\n\n"
                                "- show_symbol:\n"
                                "    Return the source code for a single symbol.\n"
                                "    Required: path, symbol\n"
                                "    Ignores: text, keep_docstring, dry_run\n\n"
                                "- replace_whole:\n"
                                "    Replace the entire definition of a symbol (class/function/variable "
                                "    assignment) with new code.\n"
                                "    Required: path, symbol, text\n"
                                "    text must contain one or more complete Python statements "
                                "    (including the 'def' or 'class' line for functions/classes).\n\n"
                                "- replace_body:\n"
                                "    Replace only the body of a function or class, keeping the original "
                                "    signature/header and (optionally) the existing docstring.\n"
                                "    Required: path, symbol, text\n"
                                "    text should contain only the body statements, NOT the 'def' or 'class' line.\n\n"
                                "- insert_before:\n"
                                "    Insert one or more statements immediately before the target symbol "
                                "    definition.\n"
                                "    Required: path, symbol, text\n"
                                "    text should be valid Python statements at the same indentation level "
                                "    as the symbol definition.\n\n"
                                "- insert_after:\n"
                                "    Insert one or more statements immediately after the target symbol "
                                "    definition.\n"
                                "    Required: path, symbol, text\n"
                                "    text should be valid Python statements at the same indentation level.\n\n"
                                "- delete_symbol:\n"
                                "    Delete the entire definition of the target symbol from the file.\n"
                                "    Required: path, symbol\n"
                                "    Ignores: text, keep_docstring.\n\n"
                                "- add_import:\n"
                                "    Add a new import statement near the top of the module (after existing "
                                "    imports or module docstring).\n"
                                "    Required: path, text\n"
                                "    text must be a valid import statement, e.g. 'import os' or "
                                "    'from pathlib import Path'.\n\n"
                                "- remove_import:\n"
                                "    Remove an existing import statement that structurally matches the "
                                "    provided import text.\n"
                                "    Required: path, text\n"
                                "    text must be the full import statement you want to remove.\n\n"
                                "- add_decorator:\n"
                                "    Add a decorator to a class or function definition.\n"
                                "    Required: path, symbol, text\n"
                                "    text must be a decorator expression like 'dataclasses.dataclass' or "
                                "    '@dataclass' (the '@' is optional).\n\n"
                                "- remove_decorator:\n"
                                "    Remove a decorator from a class or function definition.\n"
                                "    Required: path, symbol, text\n"
                                "    text should be the decorator name/expression without arguments, e.g. "
                                "    'dataclass' or 'functools.lru_cache'.\n\n"
                                "- wrap_body:\n"
                                "    Wrap the body of a function or class with additional code.\n"
                                "    Required: path, symbol, text\n"
                                "    text should contain wrapper code with exactly one 'pass' statement "
                                "    where the original body will be inserted.\n\n"
                                "- rename:\n"
                                "    Rename a symbol (class/function/method/variable) definition only.\n"
                                "    Required: path, symbol, text\n"
                                "    text must be the new identifier name (no dots).\n\n"
                                "- refactor:\n"
                                "    Rename a symbol AND all its references throughout the file.\n"
                                "    Required: path, symbol, text\n"
                                "    text must be the new identifier name (no dots).\n\n"
                                "SYMBOL FORMAT EXAMPLES:\n"
                                "- 'my_function' for a top-level function\n"
                                "- 'MyClass' for a class\n"
                                "- 'MyClass.method' for a method\n"
                                "- 'MyClass.class_attr' or 'module_variable' for assignments"
                            ),
                        },
                        "symbol": {
                            "type": "string",
                            "description": (
                                "Dot-separated name of the target symbol (class, function, method, or variable) to edit.\n\n"
                                "REQUIRED for these commands: show_symbol, replace_whole, replace_body, "
                                "insert_before, insert_after, delete_symbol, add_decorator, remove_decorator, "
                                "wrap_body, rename, refactor.\n\n"
                                "IGNORED for these commands: list_symbols, add_import, remove_import.\n\n"
                                "FORMAT EXAMPLES:\n"
                                "- 'my_function' for a top-level function\n"
                                "- 'MyClass' for a class\n"
                                "- 'MyClass.method' for a method\n"
                                "- 'MyClass.class_attr' or 'module_variable' for variable assignments"
                            ),
                        },
                        "text": {
                            "type": "string",
                            "description": (
                                "Auxiliary code or identifier text whose meaning depends on the command.\n\n"
                                "Per-command expectations:\n"
                                "- replace_whole: complete replacement statements for the symbol definition "
                                "  (including 'def'/'class' line if applicable).\n"
                                "- replace_body: body-only statements for the function/class (no header line).\n"
                                "- insert_before / insert_after: one or more complete statements to insert at the "
                                "  same indentation level as the target symbol.\n"
                                "- add_import / remove_import: a single valid import statement "
                                "  (e.g. 'import os', 'from x import y').\n"
                                "- add_decorator: decorator to add, such as 'my_decorator' or '@my_decorator'.\n"
                                "- remove_decorator: decorator name/expression to remove (no arguments), "
                                "  e.g. 'staticmethod', 'dataclass', 'cache'.\n"
                                "- wrap_body: wrapper block containing exactly one 'pass' statement where the "
                                "  original body should be spliced in (e.g. a try/except block with 'pass' in "
                                "  the try body).\n"
                                "- rename: the new short name for the symbol (identifier only, no dots).\n"
                                "- refactor: the new short name for the symbol (identifier only, no dots).\n\n"
                                "Commands that do NOT use text and will ignore it if provided: "
                                "list_symbols, show_symbol, delete_symbol."
                            ),
                        },
                        "keep_docstring": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Whether to preserve an existing docstring when replacing a function/class body.\n\n"
                                "Only meaningful for replace_body and wrap_body:\n"
                                "- If True (default): if the original target has a docstring and the new body "
                                "  does not start with a docstring, the original docstring is kept as the first "
                                "  statement of the new body.\n"
                                "- If False: the body is replaced exactly with the parsed 'text' block and any "
                                "  original docstring is discarded.\n\n"
                                "Ignored by other commands."
                            ),
                        },
                        "dry_run": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "If True, do not write changes back to disk. Instead, return a unified diff "
                                "between the original file and the edited version.\n\n"
                                "Use this for previewing changes or when you want to show a patch to the user. "
                                "For list_symbols and show_symbol the flag is ignored, because those commands "
                                "never modify the file."
                            ),
                        },
                    },
                    "required": ["command", "path"],
                },
            },
        }

    def format_output(self, data: dict) -> str:
        # Reuse similar formatting to AST editor
        lines = [f"Command: {data.get('command')}", f"Status: {data.get('status')}"]
        if "error" in data:
            lines.append(f"Error: {data['error']}")
        if "diff" in data:
            lines.append("```console")
            lines.extend(data["diff"].splitlines())
            lines.append("```")
        if "output" in data:
             lines.append(data["output"])
        return "\n".join(lines)

    async def __call__(
        self,
        *,
        command: str,
        path: str,
        symbol: Optional[str] = None,
        text: Optional[str] = None,
        keep_docstring: bool = True,
        dry_run: bool = False,
        **kwargs
    ) -> ToolResult:
        if not LIBCST_AVAILABLE:
            return ToolResult(error="LibCST is not installed. Please install it to use this tool.", tool_name=self.name)

        repo_dir = get_constant("REPO_DIR")
        if not repo_dir:
            return ToolResult(error="REPO_DIR not set", tool_name=self.name)
        
        abs_path = (Path(repo_dir) / path).resolve()
        if not _within_repo(Path(repo_dir), abs_path):
            return ToolResult(error="Path escapes REPO_DIR", tool_name=self.name)
        if not abs_path.exists():
            return ToolResult(error="File not found", tool_name=self.name)

        try:
            src = abs_path.read_text(encoding="utf-8")
            tree = cst.parse_module(src)
        except Exception as e:
            return ToolResult(error=f"Failed to parse file: {e}", tool_name=self.name)

        try:
            cmd = CSTEditorCommand(command)
        except ValueError:
             return ToolResult(error=f"Unknown command: {command}", tool_name=self.name)

        # Dispatch
        if cmd == CSTEditorCommand.LIST:
            visitor = SymbolCollector()
            tree.visit(visitor)
            output = "\n".join(f"{s['kind']:8} {s['symbol']}" for s in visitor.symbols)
            self._emit_console(str(repo_dir), f"{self.name} {command}", detail=output, path=path)
            return ToolResult(output=output, message=self.format_output({"command": command, "status": "success", "output": output}), tool_name=self.name)

        if cmd == CSTEditorCommand.SHOW:
            if not symbol: return ToolResult(error="Missing symbol", tool_name=self.name)
            # We need to find the node and print it
            # We can use a visitor to find the node, then use node.code? No, cst nodes don't have .code directly unless we use Module.code_for_node which matches by identity?
            # Actually LibCST nodes have a `code` property? No.
            # We can use `cst.Module.code_for_node(node)` provided we have the module.
            # Or `node.codegen()`? No.
            # `cst.Module` has `code_for_node`
            
            # Wait, `cst.parse_module` returns a Module. 
            # To get code for a child, we can just `child.code`? No.
            # We can use `cst.Module.code_for_node(node)`?
            # Actually, `cst.Module` has `code_for_node`.
            
            # Let's find the node first.
            finder = SymbolFinder(symbol)
            tree.visit(finder)
            if not finder.found_node:
                return ToolResult(error=f"Symbol not found: {symbol}", tool_name=self.name)
            
            # Extract code
            # LibCST nodes don't store their source code range by default unless parsed with MetadataWrapper.
            # But we can just generate code for the node: `finder.found_node.code`? No.
            # `cst.Module` object has `code` attribute.
            # We can use `cst.Module(body=[finder.found_node]).code`? That might lose indentation context.
            
            # Correct way: `cst.Module.code_for_node` is not a thing.
            # We can use `cst.metadata.PositionProvider` to get range, then slice source.
            # OR we can just `finder.found_node.code`?
            # Let's check LibCST docs memory...
            # Ah, `cst.CSTNode` does NOT have a `code` method.
            # But `cst.Module` does.
            # If we just want to show the code, we can wrap the node in a Module and print it, but indentation will be wrong?
            # Actually, `cst.Module([]).code` is empty string.
            
            # Better: `cst.parse_module("").code_for_node(node)`? No.
            
            # Let's use MetadataWrapper to get position, then slice source.
            wrapper = cst.metadata.MetadataWrapper(tree)
            # We need to re-find the node in the wrapper?
            # Or just visit the wrapper.
            
            # Let's try a simpler approach:
            # Just use `cst.Module(body=[finder.found_node]).code` and accept that indentation might be weird?
            # No, that's bad.
            
            # Let's use `cst.metadata.PositionProvider`.
            pos = wrapper.resolve(cst.metadata.PositionProvider)
            # We need to find the node instance in the wrapper's tree?
            # The wrapper wraps the tree.
            # `wrapper.visit(finder)`
            finder = SymbolFinder(symbol)
            wrapper.visit(finder)
            if not finder.found_node:
                return ToolResult(error=f"Symbol not found: {symbol}", tool_name=self.name)
            
            # Get position
            # range = pos[finder.found_node]
            # start = range.start.line
            # end = range.end.line
            # But this is line-based.
            
            # Actually, `cst.Module` has a `code` attribute which is the full source.
            # But we want just the node.
            
            # Let's fallback to `cst.Module(body=[finder.found_node]).code` for now.
            # To preserve indentation, we might need to check if it's a statement.
            node = finder.found_node
            try:
                # If node is a statement, we can wrap it in a Module
                if isinstance(node, (cst.ClassDef, cst.FunctionDef, cst.SimpleStatementLine)):
                    code = cst.Module(body=[node]).code
                elif isinstance(node, cst.BaseSmallStatement):
                    # Wrap small statement (like Assign) in a line
                    code = cst.Module(body=[cst.SimpleStatementLine(body=[node])]).code
                else:
                    # Fallback for other nodes
                    code = f"# Code generation for {type(node).__name__} not fully supported yet.\n{node}"
                
                self._emit_console(str(repo_dir), f"{self.name} {command}", detail=code, path=path, symbol=symbol)
                return ToolResult(output=code, message=self.format_output({"command": command, "status": "success", "output": code}), tool_name=self.name)
            except Exception as e:
                return ToolResult(error=f"Could not generate code for symbol: {e}", tool_name=self.name)

        # Modifications
        new_tree = tree
        if cmd in {CSTEditorCommand.ADD_IMPORT, CSTEditorCommand.REMOVE_IMPORT}:
            if not text: return ToolResult(error="Missing text", tool_name=self.name)
            transformer = ImportTransformer(cmd.value, text)
            new_tree = tree.visit(transformer)
            if not transformer.modified:
                return ToolResult(error="Import operation failed (not found or invalid)", tool_name=self.name)

        elif cmd == CSTEditorCommand.REFACTOR:
            if not symbol: return ToolResult(error="Missing symbol", tool_name=self.name)
            if not text: return ToolResult(error="Missing text (new name)", tool_name=self.name)
            
            transformer = RefactorTransformer(symbol, text)
            new_tree = tree.visit(transformer)
            if not transformer.modified:
                return ToolResult(error=f"Symbol not found or no change: {symbol}", tool_name=self.name)

        else:
            if not symbol: return ToolResult(error="Missing symbol", tool_name=self.name)
            if cmd != CSTEditorCommand.DELETE and not text: return ToolResult(error="Missing text", tool_name=self.name)
            
            transformer = EditorTransformer(cmd.value, symbol, text, keep_docstring)
            new_tree = tree.visit(transformer)
            if not transformer.modified:
                return ToolResult(error=f"Symbol not found or no change: {symbol}", tool_name=self.name)

        new_src = new_tree.code
        
        if dry_run:
            diff = _diff(src, new_src)
            self._emit_console(str(repo_dir), f"{self.name} {command} (dry-run)", detail=diff, path=path, symbol=symbol)
            return ToolResult(output=diff, message=self.format_output({"command": command, "status": "success", "diff": diff}), tool_name=self.name)
        
        # Write
        abs_path.write_text(new_src, encoding="utf-8")
        self._emit_console(str(repo_dir), f"{self.name} {command}", detail="(File updated)", path=path, symbol=symbol)
        return ToolResult(output="OK", message=self.format_output({"command": command, "status": "success"}), tool_name=self.name)
