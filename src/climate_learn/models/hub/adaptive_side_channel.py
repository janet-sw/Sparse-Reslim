"""
Side channel for adaptive sparsity diagnostics.

This is a module-level dictionary that survives FSDP wrapping.
The model writes to it during forward(), and the monitor reads from it.
"""

# Global dict for diagnostics
DIAG = {}
