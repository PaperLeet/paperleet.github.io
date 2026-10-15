"""Gradient-mode contexts/decorators for one isolated numerical runtime."""
import functools
import inspect
import sys

from . import _state


def is_grad_enabled():
    return _state._grad_enabled


class _GradContext:
    def _fresh(self):
        return type(self)()

    def __call__(self, function):
        if not callable(function):
            raise TypeError('A gradient-mode decorator requires a callable')
        if inspect.isgeneratorfunction(function):
            @functools.wraps(function)
            def generator(*args, **kwargs):
                iterator = function(*args, **kwargs)
                advance = lambda: next(iterator)
                while True:
                    try:
                        with self._fresh():
                            value = advance()
                    except StopIteration as done:
                        return done.value
                    try:
                        request = yield value
                    except GeneratorExit:
                        with self._fresh():
                            iterator.close()
                        raise
                    except BaseException:
                        error = sys.exc_info()[1]
                        advance = lambda error=error: iterator.throw(error)
                    else:
                        advance = lambda request=request: iterator.send(request)
            return generator
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with self._fresh():
                return function(*args, **kwargs)
        return wrapped


class _NoArgumentContext(_GradContext):
    def __new__(cls, function=None):
        if function is not None:
            return cls()(function)
        return super().__new__(cls)

    def __init__(self, function=None):
        self.prev = False

    def __enter__(self):
        self.prev = _state._grad_enabled
        _state._grad_enabled = self.mode

    def __exit__(self, *args):
        _state._grad_enabled = self.prev


class no_grad(_NoArgumentContext):
    mode = False


class enable_grad(_NoArgumentContext):
    mode = True


class set_grad_enabled(_GradContext):
    def __init__(self, mode):
        if type(mode) is not bool:
            raise TypeError('Gradient mode must be bool')
        self.prev, self.mode = _state._grad_enabled, mode
        _state._grad_enabled = mode

    def __enter__(self):
        _state._grad_enabled = self.mode

    def __exit__(self, *args):
        _state._grad_enabled = self.prev

    def __call__(self, function):
        _state._grad_enabled = self.prev
        return super().__call__(function)

    def _fresh(self):
        return type(self)(self.mode)
