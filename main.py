import os
import sys
from pathlib import Path
import subprocess

import model_train

from utils import ensure_dirs, get_project_paths, print_header
def install_dependencies():
    base_dir, venv_dir, data_dir, output_dir = get_project_paths()
    print_header("INSTALL DEPENDENCIES")

    # 1) create venv if not exists
    if not venv_dir.exists():
        print("[INFO] Creating virtual environment in .venv ...")
        # use the same Python that is running this script
        cmd = [sys.executable, "-m", "venv", str(venv_dir)]
        subprocess.check_call(cmd)
    else:
        print("[INFO] .venv already exists, will just install requirements.")

    # 2) pick pip inside venv
    if os.name == "nt":
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        pip_path = venv_dir / "bin" / "pip"

    if not pip_path.exists():
        raise RuntimeError("pip not found inside .venv; creation may have failed.")

    # 3) install from requirements.txt
    req_file = base_dir / "requirements.txt"
    if not req_file.exists():
        raise FileNotFoundError("requirements.txt not found next to main.py")

    print("[INFO] Installing from requirements.txt ...")
    subprocess.check_call([str(pip_path), "install", "-r", str(req_file)])

    # 4) make sure category_encoders is there (needed for target encoding in model_train)
    print("[INFO] Ensuring 'category_encoders' is installed ...")
    subprocess.check_call([str(pip_path), "install", "category_encoders"])

    print("[DONE] Dependencies installed.")


def clean_data():
    print_header("CLEAN THE DATA")
    import data_cleaning

    data_cleaning.clean_data_pipeline()


def describe_data():
    print_header("DESCRIBE THE DATA")
    import data_describe

    data_describe.describe_data()


def teach_and_test_model():
    print_header("TEACH / TRAIN & TEST THE MODEL")
    model_train.train_and_test()


def real_world_test():
    print_header("REAL WORLD TEST")
    model_train.real_world_test()


def main_menu():
    while True:
        print("\n==============================")
        print("   Land Price ML Tool")
        print("==============================")
        print("1) Install dependencies")
        print("2) Clean the data")
        print("3) Describe the data")
        print("4) Teach the model and test it")
        print("5) Real world test")
        print("6) Exit")
        choice = input("Choose an option (1-6): ").strip()

        if choice == "1":
            install_dependencies()
        elif choice == "2":
            clean_data()
        elif choice == "3":
            describe_data()
        elif choice == "4":
            teach_and_test_model()
        elif choice == "5":
            real_world_test()
        elif choice == "6":
            print("Bye.")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    # make sure directories exist
    ensure_dirs()
    _, venv_dir, _, _ = get_project_paths()
    if not venv_dir.exists():
        install_dependencies()
    main_menu()
