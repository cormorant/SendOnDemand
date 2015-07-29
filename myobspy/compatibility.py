# -*- coding: utf-8 -*-
import io
import sys

import numpy as np


PY2 = True


def frombuffer(data, dtype):
    # For compatibility with NumPy 1.4
    if isinstance(dtype, unicode):  # noqa
        dtype = str(dtype)
    if data:
        return np.frombuffer(data, dtype=dtype).copy()
    else:
        return np.array([], dtype=dtype)


def is_text_buffer(obj):
    if PY2 and not isinstance(obj, io.BufferedIOBase) and \
            not isinstance(obj, io.TextIOBase):
        if hasattr(obj, "read") and hasattr(obj, "write") \
                and hasattr(obj, "seek") and hasattr(obj, "tell"):
            return True
        return False
    return isinstance(obj, io.TextIOBase)


def is_bytes_buffer(obj):
    if PY2 and not isinstance(obj, io.BufferedIOBase) and \
            not isinstance(obj, io.TextIOBase):
        if hasattr(obj, "read") and hasattr(obj, "write") \
                and hasattr(obj, "seek") and hasattr(obj, "tell"):
            return True
        return False

    return isinstance(obj, io.BufferedIOBase)


def round_away(number):
    floor = np.floor(number)
    ceil = np.ceil(number)
    if (floor != ceil) and (abs(number - floor) == abs(ceil - number)):
        return int(int(number) + int(np.sign(number)))
    else:
        return int(np.round(number))
