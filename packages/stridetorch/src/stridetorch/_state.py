"""NumPy-backed state implementation; extracted from PaperLeet."""


_grad_enabled = True

_copy_complex_warning_emitted = False

_numpy_readonly_warning_emitted = False


_jvp_active = False


_jvp_level = 0


_storages = {}
_where_uint8_warning_emitted = False

# Functional reverse transforms reject implicit NumPy graph escapes.
_functional_ad_active = False

# HVP needs full curvature, unlike native-compatible saved-factor JVP tapes.
_hvp_active = False

_functional_jvp_active = False

_full_forward_graph_active = False

# Private HVP-only tensor pullback context; ordinary nodes retain no new fields.
_tensor_reverse_context = None
