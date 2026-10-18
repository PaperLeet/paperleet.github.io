"""NumPy-backed state implementation; extracted from PaperLeet."""


_grad_enabled = True

_copy_complex_warning_emitted = False

_numpy_readonly_warning_emitted = False


_jvp_active = False


_jvp_level = 0


_storages = {}
_where_uint8_warning_emitted = False

# Functional reverse transforms reject implicit NumPy graph escapes.
_functional_vjp_active = False
