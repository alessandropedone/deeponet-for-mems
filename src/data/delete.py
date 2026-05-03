"""
.. admonition:: Description

    This script provides a simple utility to delete a folder.
    Part of the 'data' module as a helper.


Example of usage::

    python -m data.delete --folder <data_folder_path>

"""

import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test", help="Path to the data folder to delete.")
    args = parser.parse_args()
    shutil.rmtree(args.folder, ignore_errors=True)

if __name__ == "__main__":
    main()