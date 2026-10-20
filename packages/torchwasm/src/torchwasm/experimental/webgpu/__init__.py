"""Opt-in browser matrix offload, with CPU Tensor values and gradients.

The host must attach a compatible MatrixRpc client in Pyodide before using GPU
transport. Importing this module does not create a Worker, device or torch alias.
These interfaces remain experimental; the default CPU/WASM runtime is unchanged.
"""
from .torch_bridge import MatmulOffload
from .resident_bridge import ResidentMatmulOffload
from .matrix_client import Matrix, MatrixProgram, ProgramResult
from .resources import service_sources, source_files

__all__ = ['MatmulOffload', 'ResidentMatmulOffload', 'Matrix', 'MatrixProgram',
           'ProgramResult', 'service_sources', 'source_files']
