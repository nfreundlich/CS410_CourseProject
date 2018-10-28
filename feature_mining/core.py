# -*- coding: utf-8 -*-
from . import helpers


def get_core_result():
    """Dummy feature to be returned.
    :return: dummy function
    """
    return 'core function'


def core_function():
    """Core function calling another core function
    while using a helper..."""
    if helpers.get_feature():
        print(get_core_result())
