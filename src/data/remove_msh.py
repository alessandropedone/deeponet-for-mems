"""
Script to remove all .msh files from a specified data folder.
Example of usage::

    python -m data.remove_msh --folder <data_folder_path>
"""

from .mesh import remove_msh_files
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test", help="Path to the data folder to clean msh files.")
    data_folder = parser.parse_args().folder

    remove_msh_files(data_folder=data_folder)

if __name__ == "__main__":
    main()