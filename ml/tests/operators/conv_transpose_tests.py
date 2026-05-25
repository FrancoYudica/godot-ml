import onnx
import numpy as np
from onnx import helper, TensorProto
from core import TestBuilder, TestData, build_filename
import onnxruntime as ort


def create_conv_transpose_model(model_filepath, kernel_shape, strides, dilations, pads):
    # Setup standard channel sizes
    in_channels = 1
    out_channels = 1
    
    # Weights for Transposed Conv are [In_Channels, Out_Channels/Group, kH, kW]
    w_shape = [in_channels, out_channels, kernel_shape[0], kernel_shape[1]]
    w_values = np.ones(w_shape, dtype=np.float32).flatten()
    
    weights = helper.make_tensor('W', TensorProto.FLOAT, w_shape, w_values)
    biases = helper.make_tensor('B', TensorProto.FLOAT, [out_channels], np.zeros([out_channels], dtype=np.float32))

    node = helper.make_node(
        'ConvTranspose',
        inputs=['input', 'W', 'B'],
        outputs=['output'],
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        dilations=dilations,
        name='conv_transpose_layer'
    )

    graph = helper.make_graph(
        nodes=[node],
        name='ConvTransposeModel',
        inputs=[helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', in_channels, 'height', 'width'])],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', out_channels, 'height', 'width'])],
        initializer=[weights, biases]
    )

    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    onnx.save(model, model_filepath)

class _BaseConvTranspose(TestBuilder):
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
        create_conv_transpose_model(self.model_filepath, self.kernel, self.stride, self.dilation, self.padding)


class ConvTranspose2x2(_BaseConvTranspose):
    def __init__(self, base_path):
        super().__init__(base_path, "conv_transpose_2x2", kernel=[3, 3], stride=[1, 1], pad=[1, 1, 1, 1], dilation=[1, 1])

    def create_test_data(self):
        input_shape = (1, 1, 2, 2)
        # Using your example input: [1, 2, 3, 4]
        input_data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        
        session = ort.InferenceSession(self.model_filepath, providers=['CPUExecutionProvider'])
        output = session.run(None, {'input': input_data})[0]
        
        return TestData(
            name=self.name,
            input_data=input_data.flatten().tolist(),
            input_shape=list(input_shape),
            output_data=output.flatten().tolist()
        )

class ConvTranspose4x4(_BaseConvTranspose):
    def __init__(self, base_path):
        super().__init__(base_path, "conv_transpose_4x4", kernel=[3, 3], stride=[1, 1], pad=[1, 1, 1, 1], dilation=[1, 1])

    def create_test_data(self):
        input_shape = (1, 1, 4, 4)
        # Using your example input: [1, 2, 3, 4]
        input_data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
        
        session = ort.InferenceSession(self.model_filepath, providers=['CPUExecutionProvider'])
        output = session.run(None, {'input': input_data})[0]
        
        return TestData(
            name=self.name,
            input_data=input_data.flatten().tolist(),
            input_shape=list(input_shape),
            output_data=output.flatten().tolist()
        )


class ConvTransposeDilations2(_BaseConvTranspose):
    def __init__(self, base_path):
        super().__init__(base_path, "conv_transpose_2x2", kernel=[2, 2], stride=[1, 1], pad=[0, 0, 0, 0], dilation=[2, 2])

    def create_test_data(self):
        input_shape = (1, 1, 2, 2)
        input_data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        
        session = ort.InferenceSession(self.model_filepath, providers=['CPUExecutionProvider'])
        output = session.run(None, {'input': input_data})[0]
        
        return TestData(
            name=self.name,
            input_data=input_data.flatten().tolist(),
            input_shape=list(input_shape),
            output_data=output.flatten().tolist()
        )