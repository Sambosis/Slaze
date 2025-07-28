#!/usr/bin/env python3
"""
CI integration script for the data contract system.
Orchestrates stub generation, validation, and testing.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from contract_validator import ContractValidator
from generate_stubs import StubGenerator


class ContractCI:
    """CI integration for contract-based development."""
    
    def __init__(self, contract_path: Path = Path("contract.yml"), src_dir: Path = Path("src")):
        self.contract_path = contract_path
        self.src_dir = src_dir
        self.stubs_dir = src_dir / "stubs"
        self.schemas_dir = src_dir / "schemas"
        
    def generate_stubs(self) -> bool:
        """Generate stubs from contract specification."""
        print("🔧 Generating stubs from contract specification...")
        
        try:
            generator = StubGenerator(self.contract_path, self.src_dir)
            generator.generate_all()
            return True
        except Exception as e:
            print(f"❌ Stub generation failed: {e}", file=sys.stderr)
            return False
    
    def validate_code(self, files: Optional[List[Path]] = None) -> bool:
        """Validate Python files against contract."""
        print("🔍 Validating code against contract...")
        
        if files is None:
            # Find all Python files in src directory
            files = list(self.src_dir.rglob("*.py"))
            # Exclude generated stubs and schemas
            files = [f for f in files if not str(f).startswith(str(self.stubs_dir)) and 
                    not str(f).startswith(str(self.schemas_dir))]
        
        if not files:
            print("No Python files found to validate")
            return True
        
        symbol_table_path = self.src_dir / "symbol_table.json"
        validator = ContractValidator(self.contract_path, symbol_table_path)
        
        all_valid = True
        for file_path in files:
            if file_path.exists():
                print(f"  Validating {file_path}...")
                if not validator.validate_file(file_path):
                    all_valid = False
                    print(f"  ❌ {file_path} has contract violations:")
                    for violation in validator.get_violations():
                        print(f"    • {violation}")
                else:
                    print(f"  ✅ {file_path} passes validation")
        
        if all_valid:
            print("✅ All files pass contract validation")
        else:
            print("❌ Contract validation failed")
        
        return all_valid
    
    def run_mypy(self) -> bool:
        """Run mypy type checking."""
        print("🔍 Running mypy type checking...")
        
        try:
            result = subprocess.run([
                "mypy", "--strict", str(self.src_dir)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ mypy type checking passed")
                return True
            else:
                print("❌ mypy type checking failed:")
                print(result.stdout)
                print(result.stderr)
                return False
        except FileNotFoundError:
            print("⚠️  mypy not found, skipping type checking")
            return True
        except Exception as e:
            print(f"❌ mypy execution failed: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run pytest tests."""
        print("🧪 Running tests...")
        
        try:
            result = subprocess.run([
                "pytest", "-q", "tests/"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ All tests passed")
                return True
            else:
                print("❌ Tests failed:")
                print(result.stdout)
                print(result.stderr)
                return False
        except FileNotFoundError:
            print("⚠️  pytest not found, skipping tests")
            return True
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
    
    def run_full_pipeline(self, skip_tests: bool = False) -> bool:
        """Run the complete CI pipeline."""
        print("🚀 Starting contract CI pipeline...")
        
        # Step 1: Generate stubs
        if not self.generate_stubs():
            return False
        
        # Step 2: Validate code against contract
        if not self.validate_code():
            return False
        
        # Step 3: Run mypy type checking
        if not self.run_mypy():
            return False
        
        # Step 4: Run tests (optional)
        if not skip_tests and not self.run_tests():
            return False
        
        print("🎉 Contract CI pipeline completed successfully!")
        return True
    
    def watch_and_regenerate(self):
        """Watch for contract changes and regenerate stubs."""
        print("👀 Watching for contract changes...")
        
        try:
            import time
            last_modified = self.contract_path.stat().st_mtime
            
            while True:
                time.sleep(1)
                current_modified = self.contract_path.stat().st_mtime
                
                if current_modified > last_modified:
                    print("📝 Contract changed, regenerating stubs...")
                    self.generate_stubs()
                    last_modified = current_modified
                    
        except KeyboardInterrupt:
            print("\n👋 Stopped watching")
        except Exception as e:
            print(f"❌ Watch failed: {e}")


def main():
    """Main entry point for contract CI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Contract CI pipeline")
    parser.add_argument("--contract", default="contract.yml", help="Path to contract specification")
    parser.add_argument("--src", default="src", help="Source directory")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    subparsers.add_parser("generate", help="Generate stubs from contract")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate code against contract")
    validate_parser.add_argument("files", nargs="*", help="Files to validate (default: all .py files in src)")
    
    # Type check command
    subparsers.add_parser("typecheck", help="Run mypy type checking")
    
    # Test command
    subparsers.add_parser("test", help="Run pytest tests")
    
    # Full pipeline command
    subparsers.add_parser("pipeline", help="Run full CI pipeline")
    
    # Watch command
    subparsers.add_parser("watch", help="Watch contract file and regenerate stubs")
    
    args = parser.parse_args()
    
    contract_path = Path(args.contract)
    src_dir = Path(args.src)
    
    if not contract_path.exists():
        print(f"❌ Contract specification not found: {contract_path}", file=sys.stderr)
        return 1
    
    ci = ContractCI(contract_path, src_dir)
    
    if args.command == "generate":
        success = ci.generate_stubs()
    elif args.command == "validate":
        files = [Path(f) for f in args.files] if args.files else None
        success = ci.validate_code(files)
    elif args.command == "typecheck":
        success = ci.run_mypy()
    elif args.command == "test":
        success = ci.run_tests()
    elif args.command == "pipeline":
        success = ci.run_full_pipeline(skip_tests=args.skip_tests)
    elif args.command == "watch":
        ci.watch_and_regenerate()
        return 0
    else:
        # Default to full pipeline
        success = ci.run_full_pipeline(skip_tests=args.skip_tests)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())