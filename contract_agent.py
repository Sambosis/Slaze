#!/usr/bin/env python3
"""
Agent integration for the data contract system.
Provides contract-aware code generation and validation for LLM agents.
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from contract_validator import ContractValidator


class ContractAgent:
    """Agent integration for contract-aware code generation."""
    
    def __init__(self, contract_path: Path = Path("contract.yml")):
        self.contract_path = contract_path
        self.contract = yaml.safe_load(contract_path.read_text())
        self.validator = ContractValidator(contract_path)
        
        # Load symbol table for quick reference
        symbol_table_path = Path("src/symbol_table.json")
        if symbol_table_path.exists():
            self.symbol_table = json.loads(symbol_table_path.read_text())
        else:
            self.symbol_table = self.validator.symbol_table
    
    def get_contract_prompt(self) -> str:
        """Generate a prompt section describing the contract for LLM agents."""
        prompt_parts = [
            "# DATA CONTRACT SPECIFICATION",
            "",
            "You MUST follow this data contract specification when generating code.",
            "Only use symbols, classes, functions, and types defined in this contract.",
            "Do NOT invent new parameters, rename anything, or create undefined symbols.",
            "",
            "## Contract Overview",
            f"- Module: {self.contract['module']}",
            f"- Version: {self.contract['version']}",
            f"- Description: {self.contract.get('description', 'N/A')}",
            ""
        ]
        
        # Add types section
        if 'types' in self.contract:
            prompt_parts.extend([
                "## Available Types",
                ""
            ])
            for type_name, type_spec in self.contract['types'].items():
                if type_spec['kind'] == 'dataclass':
                    fields = type_spec.get('fields', {})
                    field_list = [f"  - {name}: {type_}" for name, type_ in fields.items()]
                    prompt_parts.extend([
                        f"### {type_name}",
                        f"@dataclass with fields:",
                        *field_list,
                        ""
                    ])
        
        # Add classes section
        if 'classes' in self.contract:
            prompt_parts.extend([
                "## Available Classes",
                ""
            ])
            for class_name, class_spec in self.contract['classes'].items():
                prompt_parts.append(f"### {class_name}")
                
                # Constructor
                init_params = class_spec.get('init', {})
                if init_params:
                    param_list = [f"  - {name}: {type_}" for name, type_ in init_params.items()]
                    prompt_parts.extend([
                        "Constructor parameters:",
                        *param_list
                    ])
                
                # Attributes
                attrs = class_spec.get('attrs', {})
                if attrs:
                    attr_list = [f"  - {name}: {type_}" for name, type_ in attrs.items()]
                    prompt_parts.extend([
                        "Attributes:",
                        *attr_list
                    ])
                
                # Methods
                methods = class_spec.get('methods', {})
                if methods:
                    prompt_parts.append("Methods:")
                    for method_name, method_spec in methods.items():
                        args = method_spec.get('args', {})
                        returns = method_spec.get('returns', 'None')
                        arg_list = [f"{name}: {type_}" for name, type_ in args.items()]
                        arg_str = ", ".join(arg_list)
                        prompt_parts.append(f"  - {method_name}({arg_str}) -> {returns}")
                
                prompt_parts.append("")
        
        # Add functions section
        if 'functions' in self.contract:
            prompt_parts.extend([
                "## Available Functions",
                ""
            ])
            for func_name, func_spec in self.contract['functions'].items():
                args = func_spec.get('args', {})
                returns = func_spec.get('returns', 'None')
                arg_list = [f"{name}: {type_}" for name, type_ in args.items()]
                arg_str = ", ".join(arg_list)
                prompt_parts.append(f"- {func_name}({arg_str}) -> {returns}")
            prompt_parts.append("")
        
        # Add constants section
        if 'constants' in self.contract:
            prompt_parts.extend([
                "## Available Constants",
                ""
            ])
            for const_name, const_value in self.contract['constants'].items():
                prompt_parts.append(f"- {const_name} = {repr(const_value)}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## IMPORTANT RULES",
            "",
            "1. Import symbols ONLY from the generated stubs in src/stubs/",
            "2. Never invent new class methods, function parameters, or data fields",
            "3. Always use exact parameter names and types as specified",
            "4. Follow the exact function signatures - no additional or missing parameters",
            "5. Use only the constants defined in the contract",
            "",
            "Before generating code, verify that all symbols exist in the contract above."
        ])
        
        return "\n".join(prompt_parts)
    
    def validate_generated_code(self, code: str, filename: str = "<generated>") -> Tuple[bool, List[str]]:
        """Validate generated code against the contract."""
        is_valid = self.validator.validate_code(code, filename)
        violations = self.validator.get_violations()
        return is_valid, violations
    
    def get_retry_prompt(self, violations: List[str]) -> str:
        """Generate a retry prompt with specific violations to fix."""
        prompt_parts = [
            "# CONTRACT VIOLATION DETECTED",
            "",
            "The generated code violates the data contract. Please fix these issues:",
            ""
        ]
        
        for i, violation in enumerate(violations, 1):
            prompt_parts.append(f"{i}. {violation}")
        
        prompt_parts.extend([
            "",
            "Please regenerate the code ensuring:",
            "- All symbols are defined in the data contract",
            "- Function signatures match exactly (parameter names and types)",
            "- No new methods or attributes are added to classes",
            "- Only use imports from the contract specification",
            "",
            "Refer to the contract specification above for the correct definitions."
        ])
        
        return "\n".join(prompt_parts)
    
    def suggest_corrections(self, violations: List[str]) -> List[str]:
        """Suggest specific corrections for contract violations."""
        suggestions = []
        
        for violation in violations:
            if "not found in contract" in violation:
                # Extract symbol name
                if "Undefined symbol" in violation:
                    symbol = violation.split("'")[1]
                    similar = self._find_similar_symbols(symbol)
                    if similar:
                        suggestions.append(f"Did you mean '{similar[0]}' instead of '{symbol}'?")
                elif "Class" in violation and "not found" in violation:
                    class_name = violation.split("'")[1]
                    available_classes = list(self.contract.get('classes', {}).keys())
                    suggestions.append(f"Available classes: {', '.join(available_classes)}")
            
            elif "not defined for class" in violation:
                # Extract class and attribute names
                parts = violation.split("'")
                if len(parts) >= 4:
                    attr_name = parts[1]
                    class_name = parts[3]
                    if class_name in self.contract.get('classes', {}):
                        class_spec = self.contract['classes'][class_name]
                        available_attrs = list(class_spec.get('attrs', {}).keys())
                        available_methods = list(class_spec.get('methods', {}).keys())
                        all_available = available_attrs + available_methods
                        suggestions.append(f"Available members for {class_name}: {', '.join(all_available)}")
            
            elif "expects" in violation and "arguments" in violation:
                # Function signature mismatch
                suggestions.append("Check the function signature in the contract specification")
        
        return suggestions
    
    def _find_similar_symbols(self, symbol: str) -> List[str]:
        """Find similar symbols in the contract (simple string matching)."""
        all_symbols = set()
        
        # Collect all symbols
        all_symbols.update(self.contract.get('types', {}).keys())
        all_symbols.update(self.contract.get('classes', {}).keys())
        all_symbols.update(self.contract.get('functions', {}).keys())
        all_symbols.update(self.contract.get('constants', {}).keys())
        
        # Simple similarity check (could be improved with fuzzy matching)
        similar = []
        symbol_lower = symbol.lower()
        
        for s in all_symbols:
            if (symbol_lower in s.lower() or 
                s.lower() in symbol_lower or
                abs(len(s) - len(symbol)) <= 2):
                similar.append(s)
        
        return sorted(similar)[:3]  # Return top 3 matches
    
    def get_import_statements(self) -> List[str]:
        """Generate required import statements for contract symbols."""
        imports = [
            "# Contract-required imports",
            "from src.stubs.slazy_agent import (",
        ]
        
        # Add all types and classes
        symbols = []
        symbols.extend(self.contract.get('types', {}).keys())
        symbols.extend(self.contract.get('classes', {}).keys())
        
        for symbol in sorted(symbols):
            imports.append(f"    {symbol},")
        
        imports.extend([
            ")",
            "",
            "# Contract functions",
            "from src.stubs.slazy_agent import (",
        ])
        
        # Add functions
        functions = list(self.contract.get('functions', {}).keys())
        for func in sorted(functions):
            imports.append(f"    {func},")
        
        imports.extend([
            ")",
            "",
            "# Contract constants",
            "from src.stubs.slazy_agent import (",
        ])
        
        # Add constants
        constants = list(self.contract.get('constants', {}).keys())
        for const in sorted(constants):
            imports.append(f"    {const},")
        
        imports.append(")")
        
        return imports
    
    def create_system_prompt_section(self) -> str:
        """Create a complete system prompt section for LLM agents."""
        sections = [
            self.get_contract_prompt(),
            "",
            "## REQUIRED IMPORTS",
            "",
            "Always start your code with these imports:",
            "```python",
            *self.get_import_statements(),
            "```",
            "",
            "## CODE GENERATION WORKFLOW",
            "",
            "1. Check the contract specification above for available symbols",
            "2. Use ONLY the defined types, classes, functions, and constants",
            "3. Import symbols from src.stubs.slazy_agent",
            "4. Follow exact function signatures and parameter names",
            "5. Do not create new methods, attributes, or parameters",
            "",
            "If you're unsure about a symbol, refer back to this contract specification."
        ]
        
        return "\n".join(sections)


def main():
    """Demo the contract agent integration."""
    contract_path = Path("contract.yml")
    
    if not contract_path.exists():
        print("❌ Contract specification not found")
        return 1
    
    agent = ContractAgent(contract_path)
    
    print("=== CONTRACT PROMPT FOR LLM AGENT ===")
    print(agent.get_contract_prompt())
    print()
    
    print("=== SYSTEM PROMPT SECTION ===")
    print(agent.create_system_prompt_section())
    print()
    
    # Demo validation
    test_code = '''
from src.stubs.slazy_agent import Agent, AgentConfig

config = AgentConfig(
    model="gpt-4",
    max_tokens=1000,
    temperature=0.1,
    system_prompt="Test prompt"
)

agent = Agent(config=config, client=None)
message = agent.process_message("Hello")
'''
    
    print("=== VALIDATION DEMO ===")
    is_valid, violations = agent.validate_generated_code(test_code)
    
    if is_valid:
        print("✅ Code is valid according to contract")
    else:
        print("❌ Code has contract violations:")
        for violation in violations:
            print(f"  • {violation}")
        
        print("\n=== SUGGESTED CORRECTIONS ===")
        suggestions = agent.suggest_corrections(violations)
        for suggestion in suggestions:
            print(f"  💡 {suggestion}")
    
    return 0


if __name__ == "__main__":
    exit(main())