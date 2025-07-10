import os
from tqdm import tqdm
def find_large_dirs(root_dir, min_size_gb):
    """
    Finds directories exceeding a minimum size in gigabytes.

    Args:
        root_dir: The root directory to start the search.
        min_size_gb: The minimum directory size in gigabytes.
    """
    min_size_bytes = min_size_gb * 1024 * 1024 * 1024
    large_dirs = []

    for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
        try:
            total_size = 0
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath): #check file exists before getting size.
                    total_size += os.path.getsize(filepath)

            if total_size >= min_size_bytes:
                large_dirs.append((dirpath, total_size))
        except OSError as e:
            print(f"Error accessing {dirpath}: {e}")

    # Sort by size (largest first)
    large_dirs.sort(key=lambda x: x[1], reverse=True)

    for dirpath, size_bytes in large_dirs:
        size_gb = size_bytes / (1024 * 1024 * 1024)
        print(f"Directory: {dirpath}, Size: {size_gb:.2f} GB")

# Example usage:
root_directory = "C:\\"  # Replace with your desired root directory
minimum_size_gb = 1  # Replace with your desired minimum size in GB
find_large_dirs(root_directory, minimum_size_gb)