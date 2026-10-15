"""NumPy-backed state implementation; extracted from PaperLeet."""


_grad_enabled = True

_copy_complex_warning_emitted = False


_jvp_active = False


_jvp_level = 0


_storages = {}
