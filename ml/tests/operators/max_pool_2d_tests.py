
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from core import TestBuilder, TestData, build_filename
import numpy as np

class _BaseMaxPoolTestBuilder(TestBuilder):
    def __init__(self, base_path, kernel, stride, pad, dilation):

        name = build_filename(
            "max_pool",
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dilation
        )

        super().__init__(base_path, name)
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.dilation = dilation

    def create_model(self):
        input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', 3, 4, 4])
        output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', 3, 'h', 'w'])

        node = helper.make_node(
            'MaxPool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[self.kernel, self.kernel],
            strides=[self.stride, self.stride],
            pads=[self.pad, self.pad, self.pad, self.pad],
            dilations=[self.dilation, self.dilation]
        )

        graph = helper.make_graph([node], 'MaxPoolModel', [input_info], [output_info])
        model = helper.make_model(graph)
        model.opset_import[0].version = 13
        onnx.save(model, self.model_filepath)

    def create_test_data(self):
        input_shape = (1, 3, 4, 4)
        input_data = np.arange(1, 1 + np.prod(input_shape), dtype=np.float32).reshape(input_shape)
        
        session = ort.InferenceSession(self.model_filepath, providers=['CPUExecutionProvider'])
        output = session.run(None, {'input': input_data})[0]
        
        return TestData(
            name=self.name,
            input_data=input_data.flatten().tolist(),
            input_shape=list(input_shape),
            output_data=output.flatten().tolist()
        )

class MaxPoolTest0(_BaseMaxPoolTestBuilder):
    def __init__(self, base_path):
        super().__init__(base_path, 2, 1, 0, 1)

class MaxPoolTest1(_BaseMaxPoolTestBuilder):
    def __init__(self, base_path):
        super().__init__(base_path, 2, 1, 1, 1)

class MaxPoolTest2(_BaseMaxPoolTestBuilder):
    def __init__(self, base_path):
        super().__init__(base_path, 3, 1, 1, 1)

class MaxPoolTest3(_BaseMaxPoolTestBuilder):
    def __init__(self, base_path):
        super().__init__(base_path, 3, 2, 1, 1)

class MaxPoolTest4(_BaseMaxPoolTestBuilder):
    def __init__(self, base_path):
        super().__init__(base_path, 3, 1, 1, 2)

