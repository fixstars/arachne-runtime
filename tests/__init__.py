import pytest
import tensorflow as tf

_gpu_available = bool(tf.config.list_physical_devices("GPU"))
gpu_only = pytest.mark.skipif(not _gpu_available, reason="Need GPU")
