import onnx
import numpy as np
from onnx import helper, TensorProto
from core import build_filename, TestData, TestBuilder

class GemmIdentity(TestBuilder):

    def __init__(self, base_path):
        self.a = 1.0
        self.b = 1.0

        name = build_filename(
            "gemm_identity",
            alpha=self.a,
            beta=self.b
        )

        super().__init__(base_path, name)

    def create_model(self):
        input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['N', 3])
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['N', 3])

        weights = np.eye(3).astype(np.float32)
        bias = np.zeros(3).astype(np.float32)

        w_init = helper.make_tensor('W', TensorProto.FLOAT, [3, 3], weights.flatten())
        b_init = helper.make_tensor('B', TensorProto.FLOAT, [3], bias.flatten())

        node = helper.make_node('Gemm', ['input', 'W', 'B'], ['output'], alpha=self.a, beta=self.b, transB=1)

        graph = helper.make_graph([node], 'GemmModel', [input_info], [output_info], [w_init, b_init])
        onnx.save(helper.make_model(graph), self.model_filepath)

    def create_test_data(self):
        return TestData(
            name=self.name,
            input_data=[1.0, 2.0, 3.0],
            input_shape=[1.0, 3.0],
            output_data=[1.0, 2.0, 3.0]
        )