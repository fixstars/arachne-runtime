import numpy as np
from build import cppruntime

from arachne_runtime.module import RuntimeModuleBase, RuntimeModuleFactory


@RuntimeModuleFactory.register("cpp")
class CppRuntimeModule(RuntimeModuleBase):
    def __init__(self, **kwargs):
        print("LOG: __init__ at CppRuntimeModule")
        self.module = cppruntime.CppRuntime()

    def done(self):
        self.module.done()

    def run(self):
        print("LOG: run at CppRuntimeModule")
        self.module.run()

    def set_input(self, idx, value, **kwargs):
        print("LOG: set_input at CppRuntimeModule")
        self.module.set_input()

    def get_output(self, idx):
        print("LOG: get_output at CppRuntimeModule")
        self.module.get_output()
        return np.array([])

    def get_input_details(self):
        return []

    def get_output_details(self):
        return []

    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1):
        return
