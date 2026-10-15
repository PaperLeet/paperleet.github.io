"""Run an existing Python script with lightweight torch import aliases."""
import argparse
from pathlib import Path
import runpy
import sys

from ._integration import as_torch


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='paperleet-torch',
        description='Run a Python script with the NumPy-backed torch compatibility library.')
    parser.add_argument('script', help='Python script to execute unchanged')
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='arguments passed to the script')
    options = parser.parse_args(argv)
    script = Path(options.script).resolve()
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(script), *options.arguments]
        sys.path.insert(0, str(script.parent))
        with as_torch():
            runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
