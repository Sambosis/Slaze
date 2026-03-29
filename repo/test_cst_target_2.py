
import os

class MyClass:
    """Docstring."""
    def method(self):
        print("Original")

@existing_decorator
def my_func(x):
    return x + 1
