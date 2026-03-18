import os
import re
import sys

def reorder_by_prefix(target_prefix, target_folder):
    """
    Finds all files starting with a specific prefix (e.g., match1) in a specific folder,
    sorts them numerically, and re-indexes them starting from 0000.
    """
    # 1. SET PROJECT PATH DYNAMICALLY
    BASE_DIR = "/home/massimiliano/advertising-panel-segmentation"
    directory = os.path.join(BASE_DIR, "dataset", "images", target_folder)

    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found.")
        return

    # 2. FILTER FILES BY THE GIVEN PREFIX
    # Regex looks for: prefix_XXXX.jpg or .jpeg
    pattern = re.compile(rf"^{target_prefix}_(\d+)\.(jpg|jpeg)$", re.IGNORECASE)

    files = []
    for filename in os.listdir(directory):
        if pattern.match(filename):
            files.append(filename)

    if not files:
        print(f"No files found starting with '{target_prefix}_' in {directory}")
        return

    # 3. NUMERICAL SORT
    # Ensures order based on the existing number (e.g., _10 comes after _2)
    files.sort(key=lambda f: int(re.search(r'_(\d+)', f).group(1)))

    print(f"Found {len(files)} files for prefix '{target_prefix}' in '{target_folder}'.")
    print("Starting re-indexing...")

    # 4. SEQUENTIAL RENAME
    for index, filename in enumerate(files):
        old_path = os.path.join(directory, filename)

        # Generate new sequential name: prefix_0000.jpeg, prefix_0001.jpeg...
        new_name = f"{target_prefix}_{index:04d}.jpeg"
        new_path = os.path.join(directory, new_name)

        # Handle potential name collisions using a temporary rename
        if os.path.exists(new_path) and old_path != new_path:
            temp_path = new_path + ".tmp"
            os.rename(old_path, temp_path)
            os.rename(temp_path, new_path)
        else:
            os.rename(old_path, new_path)

        print(f"  {filename} -> {new_name}")

    print(f"\nSuccess! All '{target_prefix}' frames in '{target_folder}' are now sequential.")

if __name__ == "__main__":
    # Ensure both prefix and target folder are provided via command line
    if len(sys.argv) < 3:
        print("Usage: python3 reorder_by_prefix.py <prefix> <target_folder>")
        print("Example: python3 reorder_by_prefix.py match1 train")
        print("Example: python3 reorder_by_prefix.py match8 val")
    else:
        reorder_by_prefix(sys.argv[1], sys.argv[2])