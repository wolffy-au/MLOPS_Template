import os

def create_directory_structure(project_root):
    # Create main project directory
    os.makedirs(project_root, exist_ok=True)

    # Create subdirectories
    directories = ['data/raw', 'data/processed', 'data/external', 'notebooks', 'src/data', 'src/models',
                   'src/features', 'src/utils', 'tests', 'config', 'scripts', 'bin', 'docs']

    for directory in directories:
        os.makedirs(os.path.join(project_root, directory), exist_ok=True)

    print(f"Directory structure created in {project_root}")

if __name__ == "__main__":
    project_directory = input("Enter the path for your project root directory: ")
    create_directory_structure(project_directory)
