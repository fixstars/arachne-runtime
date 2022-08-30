# Build a C++ Runtime with the Python binding

```sh
mkdir build && cd build
cmake -G Ninja ../
ninja
```

# Import a built module from the Python code

```python
from build import cppruntime
```

# Implement a runtime plugin and use it

Please see `cpp_runtime.py` and `cpp_runtime.ipynb`.
