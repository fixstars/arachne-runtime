import os
import tempfile

import grpc
import numpy as np
import pytest
from tvm.contrib.download import download

import arachne_runtime
from arachne_runtime.rpc.server import create_server


@pytest.mark.xfail
def test_prohibit_multiple_client(rpc_port=5055):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model_path = tmp_dir + "/model.tflite"
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/mobilenet.tflite"
        download(url, model_path)

        server = create_server(rpc_port)
        server.start()

        client1 = None
        rpc_info = {"host": "localhost", "port": rpc_port}
        try:
            client1 = arachne_runtime.init(
                runtime="tflite", model_file=model_path, rpc_info=rpc_info
            )
            # cannot create multiple clients
            _ = arachne_runtime.init(runtime="tflite", model_file=model_path, rpc_info=rpc_info)

        finally:
            del client1
            server.stop(0)


def test_continue_first_client(rpc_port=5056):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model_path = tmp_dir + "/model.tflite"
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/mobilenet.tflite"
        download(url, model_path)

        dummy_input = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)  # type: ignore

        server = create_server(rpc_port)
        server.start()

        rpc_info = {"host": "localhost", "port": rpc_port}
        client1 = arachne_runtime.init(runtime="tflite", model_file=model_path, rpc_info=rpc_info)

        try:
            client1.set_input(0, dummy_input)
            # cannot create multiple clients
            _ = arachne_runtime.init(runtime="tflite", model_file=model_path, rpc_info=rpc_info)
        except grpc.RpcError:
            # client1 can continue to be used
            client1.run()
            _ = client1.get_output(0)
            del client1
        finally:
            server.stop(0)
