import os

def create_directory_structure(base_path):
    directories = [
        "config",
        "data",
        "logs",
        "models",
        "notes",
        "src",
    ]
    
    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)
        if directory == "config":
            # Create the hyper_config.yaml file

            with open(os.path.join(path, "hyper_config.json"), "w") as config:
                config.write("learning_rate: \nbatch_size: \nepochs: \n")
            print(f"Created directory: {path} and hyper_config.yaml file")
        if directory == "data":
            # Create the data directory structure
            data_directories = [
                "raw",
                "processed",
                "external",
            ]
            for data_directory in data_directories:
                data_path = os.path.join(path, data_directory)
                os.makedirs(data_path, exist_ok=True)
                print(f"Created directory: {data_path}")
            print("Local datasets go in the data/raw directory")
        if directory == "src":
            # Create the src directory structure
            src_directories = [
            "datasets",
            "model",
            "train",
            "validate",
            "visualize",
            ]
            for src_directory in src_directories:
                src_path = os.path.join(path, src_directory)
                os.makedirs(src_path, exist_ok=True)
                print(f"Created directory: {src_path}")
            
            # Create the __init__.py file
            init_file_path = os.path.join(path, "__init__.py")
            if not os.path.exists(init_file_path):
                with open(init_file_path, "w") as init_file:
                    init_file.write("# Init file for src package\n")
                print(f"Created file: {init_file_path}")
            else:
                print(f"File already exists: {init_file_path}")
            print("Put your source code in the src directory")
    # Create model_practice.py, model_test.py, README.md, and requirements.txt
    files_to_create = {
        "model_practice.py": "# Model Practice\n\n",
        "model_test.py": "# Model Test\n\n",
        "README.md": "# README\n\n",
        "requirements.txt": "# Requirements\n\n"
    }

    for file_name, file_content in files_to_create.items():
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                file.write(file_content)
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists: {file_path}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    create_directory_structure(base_path)