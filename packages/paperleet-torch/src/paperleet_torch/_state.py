"""NumPy-backed state implementation; extracted from PaperLeet."""


_grad_enabled = True


_jvp_active = False


_jvp_level = 0


_storages = {}
