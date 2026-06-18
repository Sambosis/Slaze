#!/usr/bin/env python3
"""
Simple demonstration of the data contract system.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🚀 Data Contract System Demonstration")
    print("=" * 50)
    
    print("\n📋 Contract Overview:")
    print("   • Module: slazy_agent")
    print("   • Types: 5 dataclasses defined")
    print("   • Classes: 5 agent classes defined") 
    print("   • Functions: 6 utility functions defined")
    print("   • Constants: 5 configuration constants")
    
    print("\n📁 Generated Files:")
    src_dir = Path("src")
    if src_dir.exists():
        stubs_dir = src_dir / "stubs"
        schemas_dir = src_dir / "schemas"
        
        print(f"   ✅ Stubs: {len(list(stubs_dir.glob('*.pyi')))} files" if stubs_dir.exists() else "   ❌ No stubs found")
        print(f"   ✅ Schemas: {len(list(schemas_dir.glob('*.json')))} files" if schemas_dir.exists() else "   ❌ No schemas found")
        print(f"   ✅ Implementations: {len(list(src_dir.glob('*.py')))} files")
        
        # Show a sample stub
        stub_file = stubs_dir / "slazy_agent.pyi"
        if stub_file.exists():
            print(f"\n📄 Sample from {stub_file.name}:")
            lines = stub_file.read_text().split('\n')[:15]
            for line in lines:
                print(f"   {line}")
            if len(lines) >= 15:
                print("   ... (truncated)")
    else:
        print("   ❌ No generated files found. Run 'make generate' first.")
    
    print("\n🔍 Validation Features:")
    print("   • AST-based code analysis")
    print("   • Symbol existence checking")
    print("   • Function signature validation")
    print("   • Type annotation verification")
    
    print("\n🤖 Agent Integration:")
    print("   • LLM-friendly contract prompts")
    print("   • Automatic violation detection")
    print("   • Suggested corrections")
    print("   • Retry prompt generation")
    
    print("\n⚙️  CI Integration:")
    print("   • Automatic stub generation")
    print("   • Contract validation")
    print("   • mypy type checking")
    print("   • pytest test running")
    
    print("\n📚 Usage Commands:")
    print("   make generate    - Generate stubs from contract")
    print("   make validate    - Validate code against contract")
    print("   make pipeline    - Run full CI pipeline")
    print("   make demo        - Run agent integration demo")
    print("   make clean       - Clean generated files")
    
    print("\n🎯 Key Benefits:")
    print("   ✅ Single source of truth for APIs")
    print("   ✅ Automatic code generation")
    print("   ✅ Real-time validation")
    print("   ✅ LLM agent integration")
    print("   ✅ CI/CD pipeline support")
    
    print("\n🎉 System is ready! Run 'make help' for available commands.")

if __name__ == "__main__":
    main()