import onnx
import numpy as np
from onnx import helper, TensorProto
from core import TestBuilder, TestData, build_filename
import onnxruntime as ort

def create_model(model_filepath, kernel_shape, strides, dilations, pads):
    # Model parameters
    in_channels = 3
    out_channels = 8
    
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', in_channels, 'height', 'width'])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', out_channels, 'height', 'width'])
    w_shape = [out_channels, in_channels, kernel_shape[0], kernel_shape[1]]
    w_values = np.ones(w_shape).astype(np.float32).flatten()
    
    weights = helper.make_tensor(
        name='W',
        data_type=TensorProto.FLOAT,
        dims=w_shape,
        vals=w_values
    )

    # Empty bias
    b_shape = [out_channels]
    b_values = np.zeros(b_shape).astype(np.float32).flatten()
    
    biases = helper.make_tensor(
        name='B',
        data_type=TensorProto.FLOAT,
        dims=b_shape,
        vals=b_values
    )

    node = helper.make_node(
        'Conv',
        inputs=['input', 'W', 'B'],
        outputs=['output'],
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        dilations=dilations,
        name='conv_layer'
    )

    graph = helper.make_graph(
        nodes=[node],
        name='Conv2DModel',
        inputs=[input_info],
        outputs=[output_info],
        initializer=[weights, biases]
    )

    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    onnx.save(model, model_filepath)

class _BaseConv(TestBuilder):
    def __init__(self, base_path, name, kernel, stride, pad, dilation):
        name = build_filename(
            name,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dilation
        )
        super().__init__(base_path, name)

        self.kernel = kernel
        self.stride = stride
        self.padding = pad
        self.dilation = dilation

    def create_model(self):
        create_model(self.model_filepath, self.kernel, self.stride, self.dilation, self.padding)



class Conv2DTest1(_BaseConv):

    def __init__(self, base_path):
        super().__init__(base_path, "conv2d", kernel=[3, 3], stride=[2, 2], pad=[1, 1, 1, 1], dilation=[1, 1])

    def create_test_data(self):
        input_shape=(1, 3, 3, 3)
        input_data = np.ones(input_shape, dtype=np.float32)
        session = ort.InferenceSession(self.model_filepath, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        output_data = outputs[0].flatten().tolist()
        return TestData(
            name=self.name,
            input_data=input_data.flatten().tolist(),
            input_shape=list(input_shape),
            output_data=output_data
        )