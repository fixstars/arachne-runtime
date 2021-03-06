import os
import tempfile

import numpy as np
import onnxruntime as ort
from tests import gpu_only
from tvm.contrib.download import download

import arachne_runtime


def _test_onnx_runtime(providers):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        url = (
            "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/resnet18.onnx"
        )

        onnx_model_path = tmp_dir + "/resnet18.onnx"
        download(url, onnx_model_path)
        input_data = np.array(np.random.random_sample([1, 3, 224, 224]), dtype=np.float32)  # type: ignore

        # ONNX Runtime
        sess = ort.InferenceSession(onnx_model_path, providers=providers)
        input_name = sess.get_inputs()[0].name
        dout = sess.run(output_names=None, input_feed={input_name: input_data})[0]
        del sess

        # Arachne Runtime
        ort_opts = {"providers": providers}
        runtime_module = arachne_runtime.init(
            runtime="onnx", model_file=onnx_model_path, **ort_opts
        )
        runtime_module.set_input(0, input_data)
        runtime_module.run()
        aout = runtime_module.get_output(0)

        if "CPUExecutionProvider" in providers:
            np.testing.assert_equal(actual=aout, desired=dout)
        else:
            np.testing.assert_allclose(actual=aout, desired=dout)
        runtime_module.benchmark()


def test_onnx_runtime_cpu():
    _test_onnx_runtime(["CPUExecutionProvider"])


@gpu_only
def test_onnx_runtime_gpu():
    _test_onnx_runtime(["CUDAExecutionProvider"])
