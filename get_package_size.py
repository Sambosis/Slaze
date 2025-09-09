
import json
import subprocess
import os
from importlib import import_module
from datetime import datetime

def get_package_size(package_name):
    try:
        module = import_module(package_name)
        package_path = os.path.dirname(module.__file__)
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(package_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
    except Exception as e:
        return f"Error: {str(e)}"

def get_pip_packages():
    result = subprocess.run(['pip', 'list', '--format=json'], capture_output=True, text=True)
    return json.loads(result.stdout)

def main():
    packages = get_pip_packages()
    package_sizes = []

    for package in packages:
        name = package['name']
        size = get_package_size(name)
        package_sizes.append({'name': name, 'version': package['version'], 'size_bytes': size})

    # Sort by size in descending order
    package_sizes.sort(key=lambda x: x['size_bytes'] if isinstance(x['size_bytes'], int) else -1, reverse=True)

    # Print or save the results
    print(json.dumps(package_sizes, indent=2))

if __name__ == "__main__":
    main()
