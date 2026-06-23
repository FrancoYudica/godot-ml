import onnx
import math
from onnx import helper, TensorProto
from core import TestBuilder, TestData

class SigmoidBasic(TestBuilder):

    def __init__(self, base_path):
        super().__init__(base_path, "sigmoid_basic")

    def create_model(self):
        input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['N', 3])
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['N', 3])
        node = helper.make_node('Sigmoid', ['input'], ['output'])
        graph = helper.make_graph([node], 'SigmoidModel', [input_info], [output_info], [])
        onnx.save(helper.make_model(graph), self.model_filepath)


    def create_test_data(self):
        input_data = [-1.0, 0.0, 1.0]
        output_data = [
            1.0 / (1.0 + math.exp(-x)) for x in input_data
        ]

        return TestData(
            name=self.name,
            input_data=input_data,
            input_shape=[1.0, 3.0],
            output_data=output_data
        )