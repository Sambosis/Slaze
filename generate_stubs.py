#!/usr/bin/env python3
"""
Stub generator for the data contract system.
Converts YAML contract specification into Python stubs, dataclasses, and JSON schema.
"""

import ast
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class StubGenerator:
    """Generates Python stubs from YAML contract specification."""
    
    def __init__(self, spec_path: Path, output_dir: Path):
        self.spec_path = spec_path
        self.output_dir = output_dir
        self.stubs_dir = output_dir / "stubs"
        self.schema_dir = output_dir / "schemas"
        
        # Create output directories
        self.stubs_dir.mkdir(parents=True, exist_ok=True)
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and parse spec
        self.spec = yaml.safe_load(spec_path.read_text())
        self.symbol_table: Dict[str, Dict[str, Any]] = {}
        
    def _parse_type_annotation(self, type_str: str) -> str:
        """Parse type annotation, handling optional types and generics."""
        if type_str.startswith('"') and type_str.endswith('"'):
            # Remove quotes from optional types
            inner_type = type_str[1:-1]
            if inner_type.endswith('?'):
                base_type = inner_type[:-1]
                return f"Optional[{base_type}]"
            return inner_type
        elif type_str.endswith('?'):
            # Handle unquoted optional types
            base_type = type_str[:-1]
            return f"Optional[{base_type}]"
        
        # Handle forward references for custom types
        if 'list[' in type_str and ']' in type_str:
            # Extract the type inside list[]
            start = type_str.find('[') + 1
            end = type_str.rfind(']')
            inner_type = type_str[start:end]
            if inner_type in self.spec.get('types', {}):
                return f'list["{inner_type}"]'
        
        return type_str
    
    def _generate_dataclass_stub(self, name: str, spec: Dict[str, Any]) -> str:
        """Generate a dataclass stub for .pyi files."""
        fields = spec.get('fields', {})
        
        field_lines = []
        for field_name, field_type in fields.items():
            parsed_type = self._parse_type_annotation(field_type)
            field_lines.append(f"    {field_name}: {parsed_type}")
        
        fields_block = '\n'.join(field_lines) if field_lines else "    pass"
        
        return f"@dataclass\nclass {name}:\n{fields_block}"
    
    def _generate_dataclass(self, name: str, spec: Dict[str, Any]) -> str:
        """Generate a dataclass from type specification."""
        fields = spec.get('fields', {})
        
        imports = {'from dataclasses import dataclass'}
        field_lines = []
        
        has_optional = any(
            field_type.startswith('"') and field_type.endswith('?"')
            for field_type in fields.values()
        )
        
        if has_optional or any('Optional[' in self._parse_type_annotation(ft) for ft in fields.values()):
            imports.add('from typing import Optional')
        
        # Check for generic types
        for field_type in fields.values():
            if 'list[' in field_type or 'dict[' in field_type:
                imports.add('from typing import List, Dict')
                break
        
        for field_name, field_type in fields.items():
            parsed_type = self._parse_type_annotation(field_type)
            field_lines.append(f"{field_name}: {parsed_type}")
        
        import_block = '\n'.join(sorted(imports))
        fields_block = '\n'.join(field_lines) if field_lines else "pass"
        
        class_def = f"@dataclass\nclass {name}:\n"
        if field_lines:
            for line in field_lines:
                class_def += f"    {line}\n"
        else:
            class_def += "    pass\n"
        
        return f"{import_block}\n\n{class_def}".strip()
    
    def _generate_class_stub(self, name: str, spec: Dict[str, Any]) -> str:
        """Generate a class stub from class specification."""
        init_params = spec.get('init', {})
        attrs = spec.get('attrs', {})
        methods = spec.get('methods', {})
        extends = spec.get('extends')
        
        imports = set()
        if any('Optional[' in str(v) for v in {**init_params, **attrs}.values()):
            imports.add('from typing import Optional')
        
        # Generate __init__ method
        init_lines = []
        if init_params:
            params = []
            for param_name, param_type in init_params.items():
                parsed_type = self._parse_type_annotation(param_type)
                params.append(f"{param_name}: {parsed_type}")
            
            init_signature = f"def __init__(self, {', '.join(params)}) -> None: ..."
            init_lines.append(f"    {init_signature}")
        
        # Generate attributes
        attr_lines = []
        for attr_name, attr_type in attrs.items():
            parsed_type = self._parse_type_annotation(attr_type)
            attr_lines.append(f"    {attr_name}: {parsed_type}")
        
        # Generate methods
        method_lines = []
        for method_name, method_spec in methods.items():
            args = method_spec.get('args', {})
            returns = method_spec.get('returns', 'None')
            
            params = []
            for arg_name, arg_type in args.items():
                parsed_type = self._parse_type_annotation(arg_type)
                params.append(f"{arg_name}: {parsed_type}")
            
            param_str = ', '.join(params)
            if param_str:
                param_str = ', ' + param_str
            
            parsed_return = self._parse_type_annotation(returns)
            method_lines.append(f"    def {method_name}(self{param_str}) -> {parsed_return}: ...")
        
        # Build class definition
        base_class = f"({extends})" if extends else ""
        class_def = f"class {name}{base_class}:"
        
        body_lines = []
        if attr_lines:
            body_lines.extend(attr_lines)
        if init_lines:
            body_lines.extend(init_lines)
        if method_lines:
            body_lines.extend(method_lines)
        
        if not body_lines:
            body_lines = ["    pass"]
        
        import_block = '\n'.join(sorted(imports)) + '\n' if imports else ''
        body_block = '\n'.join(body_lines)
        
        return f"{import_block}\n{class_def}\n{body_block}"
    
    def _generate_function_stub(self, name: str, spec: Dict[str, Any]) -> str:
        """Generate a function stub from function specification."""
        args = spec.get('args', {})
        returns = spec.get('returns', 'None')
        
        params = []
        for arg_name, arg_type in args.items():
            parsed_type = self._parse_type_annotation(arg_type)
            params.append(f"{arg_name}: {parsed_type}")
        
        param_str = ', '.join(params)
        parsed_return = self._parse_type_annotation(returns)
        
        return f"def {name}({param_str}) -> {parsed_return}: ..."
    
    def _generate_constants(self, constants: Dict[str, Any]) -> str:
        """Generate constants module."""
        lines = []
        for name, value in constants.items():
            if isinstance(value, str):
                lines.append(f'{name}: str = "{value}"')
            elif isinstance(value, (int, float)):
                lines.append(f'{name}: {type(value).__name__} = {value}')
            else:
                lines.append(f'{name} = {repr(value)}')
        
        return '\n'.join(lines)
    
    def _generate_json_schema(self) -> Dict[str, Any]:
        """Generate JSON schema from the contract specification."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": f"{self.spec['module']} API Contract",
            "description": self.spec.get('description', ''),
            "version": self.spec['version'],
            "definitions": {}
        }
        
        # Add type definitions
        for type_name, type_spec in self.spec.get('types', {}).items():
            if type_spec['kind'] == 'dataclass':
                properties = {}
                required = []
                
                for field_name, field_type in type_spec['fields'].items():
                    if field_type.startswith('"') and field_type.endswith('?"'):
                        # Optional field
                        base_type = field_type[1:-2]
                        properties[field_name] = self._type_to_json_schema(base_type)
                    else:
                        properties[field_name] = self._type_to_json_schema(field_type)
                        required.append(field_name)
                
                schema['definitions'][type_name] = {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
        
        return schema
    
    def _type_to_json_schema(self, type_str: str) -> Dict[str, Any]:
        """Convert Python type annotation to JSON schema type."""
        type_mapping = {
            'str': {"type": "string"},
            'int': {"type": "integer"},
            'float': {"type": "number"},
            'bool': {"type": "boolean"},
            'dict': {"type": "object"},
            'object': {"type": "object"}
        }
        
        if type_str in type_mapping:
            return type_mapping[type_str]
        elif type_str.startswith('list['):
            inner_type = type_str[5:-1]
            return {
                "type": "array",
                "items": self._type_to_json_schema(inner_type)
            }
        else:
            # Assume it's a custom type reference
            return {"$ref": f"#/definitions/{type_str}"}
    
    def _build_symbol_table(self):
        """Build symbol table for validation."""
        # Add types
        for type_name, type_spec in self.spec.get('types', {}).items():
            self.symbol_table[type_name] = {
                'kind': 'type',
                'fields': type_spec.get('fields', {}),
                'spec': type_spec
            }
        
        # Add classes
        for class_name, class_spec in self.spec.get('classes', {}).items():
            methods = {}
            for method_name, method_spec in class_spec.get('methods', {}).items():
                methods[method_name] = {
                    'args': method_spec.get('args', {}),
                    'returns': method_spec.get('returns', 'None')
                }
            
            self.symbol_table[class_name] = {
                'kind': 'class',
                'init': class_spec.get('init', {}),
                'attrs': class_spec.get('attrs', {}),
                'methods': methods,
                'spec': class_spec
            }
        
        # Add functions
        for func_name, func_spec in self.spec.get('functions', {}).items():
            self.symbol_table[func_name] = {
                'kind': 'function',
                'args': func_spec.get('args', {}),
                'returns': func_spec.get('returns', 'None'),
                'spec': func_spec
            }
    
    def generate_all(self):
        """Generate all artifacts from the contract specification."""
        self._build_symbol_table()
        
        # Generate type stubs (.pyi files)
        self._generate_type_stubs()
        
        # Generate implementation shells
        self._generate_implementation_shells()
        
        # Generate JSON schema
        self._generate_schema_files()
        
        # Generate symbol table for validation
        self._generate_symbol_table_file()
        
        print(f"✓ Generated stubs in {self.stubs_dir}")
        print(f"✓ Generated implementations in {self.output_dir}")
        print(f"✓ Generated schemas in {self.schema_dir}")
    
    def _generate_type_stubs(self):
        """Generate .pyi stub files."""
        # Main module stub
        main_stub_lines = []
        
        # Add imports
        main_stub_lines.extend([
            "from typing import Optional, List, Dict, Any",
            "from dataclasses import dataclass",
            ""
        ])
        
        # Add types
        for type_name, type_spec in self.spec.get('types', {}).items():
            if type_spec['kind'] == 'dataclass':
                stub_code = self._generate_dataclass_stub(type_name, type_spec)
                main_stub_lines.append(stub_code)
                main_stub_lines.append("")
        
        # Add classes
        for class_name, class_spec in self.spec.get('classes', {}).items():
            stub_code = self._generate_class_stub(class_name, class_spec)
            main_stub_lines.append(stub_code)
            main_stub_lines.append("")
        
        # Add functions
        for func_name, func_spec in self.spec.get('functions', {}).items():
            stub_code = self._generate_function_stub(func_name, func_spec)
            main_stub_lines.append(stub_code)
            main_stub_lines.append("")
        
        # Add constants
        if 'constants' in self.spec:
            constants_code = self._generate_constants(self.spec['constants'])
            main_stub_lines.append(constants_code)
        
        # Write main stub file
        main_stub_content = '\n'.join(main_stub_lines)
        (self.stubs_dir / f"{self.spec['module']}.pyi").write_text(main_stub_content)
    
    def _generate_implementation_shells(self):
        """Generate implementation shell files."""
        # Generate dataclass implementations
        for type_name, type_spec in self.spec.get('types', {}).items():
            if type_spec['kind'] == 'dataclass':
                impl_code = self._generate_dataclass(type_name, type_spec)
                (self.output_dir / f"{type_name.lower()}.py").write_text(impl_code)
        
        # Generate constants file
        if 'constants' in self.spec:
            constants_code = self._generate_constants(self.spec['constants'])
            (self.output_dir / "constants.py").write_text(constants_code)
    
    def _generate_schema_files(self):
        """Generate JSON schema files."""
        schema = self._generate_json_schema()
        schema_file = self.schema_dir / f"{self.spec['module']}_schema.json"
        schema_file.write_text(json.dumps(schema, indent=2))
    
    def _generate_symbol_table_file(self):
        """Generate symbol table file for validation."""
        symbol_table_file = self.output_dir / "symbol_table.json"
        symbol_table_file.write_text(json.dumps(self.symbol_table, indent=2))


def main():
    """Main entry point for stub generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate stubs from contract specification")
    parser.add_argument("--spec", default="contract.yml", help="Path to contract specification")
    parser.add_argument("--output", default="src", help="Output directory")
    
    args = parser.parse_args()
    
    spec_path = Path(args.spec)
    output_dir = Path(args.output)
    
    if not spec_path.exists():
        print(f"Error: Contract specification not found: {spec_path}")
        return 1
    
    generator = StubGenerator(spec_path, output_dir)
    generator.generate_all()
    
    return 0


if __name__ == "__main__":
    exit(main())