import numpy as np

from arachne_runtime.module import RuntimeModuleBase, RuntimeModuleFactory


@RuntimeModuleFactory.register("simple")
class SimpleRuntimeModule(RuntimeModuleBase):
    def __init__(self, **kwargs):
        print("LOG: __init__ at SimpleRuntimeModule")

    def run(self):
        print("LOG: run at SimpleRuntimeModule")

    def set_input(self, idx, value, **kwargs):
        """Set input data.

        Args:
            idx (int): layer index to set data
            value (np.ndarray): input data
        """
        print("LOG: set_input at SimpleRuntimeModule")

    def get_output(self, idx):
        print("LOG: get_output at SimpleRuntimeModule")
        return np.array([])

    def get_input_details(self):
        print("LOG: get_input_details at SimpleRuntimeModule")
        return []

    def get_output_details(self):
        print("LOG: get_output_details at SimpleRuntimeModule")
        return []

    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1):
        print("LOG: benchmark at SimpleRuntimeModule")
        return
