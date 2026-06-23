import onnx
from onnx import helper, TensorProto
from core import TestBuilder, TestData

class ReLuBasic(TestBuilder):

    def __init__(self, base_path):
        super().__init__(base_path, "relu_basic")

    def create_model(self):
        input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['N', 3])
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['N', 3])
        node = helper.make_node('Relu', ['input'], ['output'])
        graph = helper.make_graph([node], 'ReluModel', [input_info], [output_info], [])
        onnx.save(helper.make_model(graph), self.model_filepath)


    def create_test_data(self):
        return TestData(
            name=self.name,
            input_data=[-1.0, 0.0, 1.0],
            input_shape=[1.0, 3.0],
            output_data=[0.0, 0.0, 1.0]
        )