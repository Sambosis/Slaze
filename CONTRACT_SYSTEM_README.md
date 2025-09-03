# Data Contract System

A comprehensive system for ensuring consistent APIs across all agent-generated code through a single, authoritative data contract specification.

## Overview

This system implements the following workflow:

1. **Single Contract Source** - All APIs defined in `contract.yml`
2. **Stub Generation** - Automatic generation of Python stubs and type definitions
3. **Code Validation** - AST-based validation against the contract
4. **CI Integration** - Automated pipeline with mypy and pytest
5. **Agent Integration** - LLM-friendly prompts and validation

## Quick Start

### 1. Install Dependencies

```bash
make install
# or manually:
pip install pyyaml mypy pytest
```

### 2. Generate Stubs

```bash
make generate
```

This creates:
- `src/stubs/` - Python stub files (.pyi)
- `src/schemas/` - JSON schemas for validation
- `src/symbol_table.json` - Symbol table for validation
- `src/*.py` - Dataclass implementations

### 3. Validate Code

```bash
make validate
```

### 4. Run Full Pipeline

```bash
make pipeline
```

## Contract Specification Format

The `contract.yml` file defines all types, classes, functions, and constants:

```yaml
version: 1
module: my_module
description: "API contract for my module"

types:
  MyData:
    kind: dataclass
    fields:
      name: str
      value: int
      optional_field: "str?"

classes:
  MyClass:
    init:
      param1: str
      param2: int
    attrs:
      property1: str
    methods:
      my_method:
        args:
          arg1: str
        returns: MyData

functions:
  my_function:
    args:
      input: str
    returns: bool

constants:
  DEFAULT_VALUE: "hello"
  MAX_SIZE: 100
```

### Contract Rules

1. **Single Source** - One contract file per bounded context
2. **Stable IDs** - Never rename keys; deprecate instead
3. **Explicit Cardinality** - Use `list[Type]` for arrays, `"Type?"` for optional
4. **No Default Types** - Always specify types explicitly

## Generated Artifacts

### Python Stubs (.pyi files)

```python
# src/stubs/my_module.pyi
from typing import Optional
from dataclasses import dataclass

@dataclass
class MyData:
    name: str
    value: int
    optional_field: Optional[str]

class MyClass:
    property1: str
    def __init__(self, param1: str, param2: int) -> None: ...
    def my_method(self, arg1: str) -> MyData: ...

def my_function(input: str) -> bool: ...
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "my_module API Contract",
  "definitions": {
    "MyData": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "value": {"type": "integer"},
        "optional_field": {"type": "string"}
      },
      "required": ["name", "value"]
    }
  }
}
```

## Validation

The system performs comprehensive validation:

### AST-Based Validation

- **Symbol Existence** - All referenced symbols must exist in contract
- **Function Signatures** - Parameter names and types must match exactly
- **Class Members** - Only defined attributes and methods allowed
- **Call Arguments** - Function calls must match expected signatures

### Example Violations

```python
# ❌ Undefined symbol
result = UndefinedClass()

# ❌ Wrong parameter name
my_function(wrong_param="value")  # Should be 'input'

# ❌ Undefined method
obj.undefined_method()

# ❌ Wrong argument count
my_function("arg1", "extra_arg")
```

## Agent Integration

### System Prompt Integration

```python
from contract_agent import ContractAgent

agent = ContractAgent()
system_prompt = agent.create_system_prompt_section()
```

This generates a comprehensive prompt section including:
- Available types, classes, and functions
- Required imports
- Validation rules
- Code generation guidelines

### Validation Workflow

```python
# Validate generated code
is_valid, violations = agent.validate_generated_code(code)

if not is_valid:
    # Get retry prompt with specific violations
    retry_prompt = agent.get_retry_prompt(violations)
    
    # Get suggested corrections
    suggestions = agent.suggest_corrections(violations)
```

## CI Integration

### Automated Pipeline

```bash
# Full pipeline
make pipeline

# Individual steps
make generate    # Generate stubs
make validate    # Validate code
make typecheck   # Run mypy
make test        # Run pytest
```

### Watch Mode

```bash
make watch
```

Automatically regenerates stubs when the contract file changes.

### CI Configuration

Add to your CI pipeline:

```yaml
# .github/workflows/contract.yml
name: Contract Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: make install
      - name: Run contract pipeline
        run: make pipeline
```

## Command Reference

### Make Commands

| Command | Description |
|---------|-------------|
| `make help` | Show available commands |
| `make install` | Install dependencies |
| `make generate` | Generate stubs from contract |
| `make validate` | Validate code against contract |
| `make typecheck` | Run mypy type checking |
| `make test` | Run pytest tests |
| `make pipeline` | Run full CI pipeline |
| `make watch` | Watch contract file for changes |
| `make demo` | Run contract agent demo |
| `make clean` | Clean generated files |

### Python Scripts

| Script | Description |
|--------|-------------|
| `generate_stubs.py` | Generate stubs from contract |
| `contract_validator.py` | Validate code against contract |
| `contract_ci.py` | CI pipeline orchestration |
| `contract_agent.py` | Agent integration utilities |

### Script Usage

```bash
# Generate stubs
python generate_stubs.py --spec contract.yml --output src

# Validate files
python contract_validator.py --contract contract.yml file1.py file2.py

# Run CI pipeline
python contract_ci.py pipeline

# Watch for changes
python contract_ci.py watch
```

## Best Practices

### Contract Design

1. **Start Small** - Begin with core types and expand incrementally
2. **Version Carefully** - Never break existing contracts
3. **Document Changes** - Use descriptive commit messages for contract changes
4. **Review Thoroughly** - Contract changes affect all generated code

### Code Generation

1. **Import First** - Always import from generated stubs
2. **Validate Early** - Run validation before committing code
3. **Fix Violations** - Address contract violations immediately
4. **Use Constants** - Reference contract constants instead of hardcoding

### Development Workflow

1. **Update Contract** - Modify `contract.yml` first
2. **Generate Stubs** - Run `make generate`
3. **Write Code** - Use generated stubs for imports
4. **Validate** - Run `make validate`
5. **Test** - Run `make test`
6. **Commit** - Commit contract and generated files together

## Troubleshooting

### Common Issues

**Q: "Symbol not found in contract" error**
A: Check that the symbol is defined in `contract.yml` and regenerate stubs.

**Q: "Function signature mismatch" error**
A: Ensure parameter names and types match the contract exactly.

**Q: "mypy errors after contract change"**
A: Regenerate stubs with `make generate` and update imports.

**Q: "Generated stubs are outdated"**
A: Run `make clean && make generate` to regenerate from scratch.

### Debug Mode

Enable verbose validation:

```bash
python contract_validator.py --contract contract.yml --verbose file.py
```

### Contract Validation

Validate the contract itself:

```bash
python -c "import yaml; yaml.safe_load(open('contract.yml'))"
```

## Architecture

```
contract.yml          # Single source of truth
     ↓
generate_stubs.py     # Generates artifacts
     ↓
├── stubs/           # Python stubs (.pyi)
├── schemas/         # JSON schemas
└── symbol_table.json # Validation data
     ↓
contract_validator.py # AST-based validation
     ↓
contract_ci.py       # CI orchestration
     ↓
contract_agent.py    # Agent integration
```

## Extension Points

### Custom Validators

Add custom validation rules by extending `ASTValidator`:

```python
class CustomValidator(ASTValidator):
    def visit_FunctionDef(self, node):
        # Custom validation logic
        super().visit_FunctionDef(node)
```

### Additional Generators

Create new generators by extending `StubGenerator`:

```python
class CustomGenerator(StubGenerator):
    def generate_custom_artifacts(self):
        # Generate additional artifacts
        pass
```

### Integration Hooks

Add pre/post-generation hooks:

```python
def pre_generation_hook(contract):
    # Custom logic before generation
    pass

def post_generation_hook(artifacts):
    # Custom logic after generation
    pass
```

## License

This data contract system is part of the Slazy project and follows the same licensing terms.