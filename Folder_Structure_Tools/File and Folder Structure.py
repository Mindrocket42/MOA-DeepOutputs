import os

def save_folder_structure(root_folder, output_file):
    """Traverse the directory and save the folder structure to a text file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        for folder_path, subfolders, filenames in os.walk(root_folder):
            indent_level = folder_path.replace(root_folder, '').count(os.sep)
            indent = '│   ' * indent_level  # Create tree structure indentation
            folder_name = os.path.basename(folder_path)

            file.write(f"{indent}├── {folder_name}/\n")

            for filename in filenames:
                file.write(f"{indent}│   ├── {filename}\n")

def main():
    root_folder = input("Enter the root folder path: ").strip()

    if not os.path.exists(root_folder):
        print("Error: The specified folder does not exist.")
        return

    output_file = "folder_structure.txt"
    save_folder_structure(root_folder, output_file)
    print(f"Folder structure saved to '{output_file}'")

if __name__ == "__main__":
    main()