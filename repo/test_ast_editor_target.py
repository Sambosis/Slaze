import os

GLOBAL_VAR = 123

class MyClass:
    def my_method(self):
        pass

def outer_func():
    def inner_func():
        return "inner"
    return inner_func
