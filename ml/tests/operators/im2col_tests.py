import onnx
from onnx import helper, TensorProto
from core import build_filename, TestData, TestBuilder
import numpy as np

def im2col_evaluate(input_data, input_shape, kernel_shape, strides, pads, dilations):
    # Dimensions [1, 1, H, W]
    h_in, w_in = input_shape[2], input_shape[3]
    k_h, k_w = kernel_shape
    s_h, s_w = strides
    p_t, p_l, p_b, p_r = pads
    d_h, d_w = dilations

    # Output Dimensions
    h_out = (h_in + p_t + p_b - (d_h * (k_h - 1) + 1)) // s_h + 1
    w_out = (w_in + p_l + p_r - (d_w * (k_w - 1) + 1)) // s_w + 1

    output = []

    # 3. Slide over output patches
    for y in range(h_out):
        for x in range(w_out):
            # 4. Extract patch
            for ky in range(k_h):
                for kx in range(k_w):
                    # Calculate input indices with dilation
                    iy = y * s_h + ky * d_h - p_t
                    ix = x * s_w + kx * d_w - p_l
                    
                    # 5. Boundary check (Padding)
                    if 0 <= iy < h_in and 0 <= ix < w_in:
                        # Convert 2D index to 1D index
                        idx = iy * w_in + ix
                        output.append(float(input_data[idx]))
                    else:
                        # Padding value is 0
                        output.append(0.0)
    return output

class _BaseIm2Col(TestBuilder):
    def __init__(self, base_path, kernel, stride, pad, dilation):

        name = build_filename(
            "im2col",
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
        # We define a dummy graph just to satisfy your engine's loader, 
        # but the ONNX runtime won't actually run this.
        input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 3, 3])
        node = helper.make_node(
            'Im2Col', inputs=['input'], outputs=['output'],
            kernel_shape=[self.kernel, self.kernel],
            strides=[self.stride, self.stride],
            pads=[self.pad, self.pad, self.pad, self.pad],
            dilations=[self.dilation, self.dilation]
        )
        graph = helper.make_graph([node], 'Im2ColModel', [input_info], 
                                  [helper.make_tensor_value_info('output', TensorProto.FLOAT, None)])
        model = helper.make_model(graph)
        onnx.save(model, self.model_filepath)

    def create_input_data(self):
        pass

    def create_output_data(self):
        pass

    def create_test_data(self):
        pass


class _BaseIm2Col3x3(_BaseIm2Col):
    def create_input_data(self):
        return [
            1, 2, 3, 
            4, 5, 6, 
            7, 8, 9
        ]
    def create_output_data(self):
        return im2col_evaluate(
            self.create_input_data(),
            [1, 1, 3, 3],
            [self.kernel, self.kernel],
            [self.stride, self.stride],
            [self.pad, self.pad, self.pad, self.pad],
            [self.dilation, self.dilation]
        )

    def create_test_data(self):
        return TestData(
            name=self.name,
            input_data=self.create_input_data(),
            input_shape=[1, 1, 3, 3],
            output_data=self.create_output_data()
        )

class _BaseIm2Col4x4(_BaseIm2Col):
    def create_input_data(self):
        return [
            1, 2, 3, 4, 
            5, 6, 7, 8, 
            9, 10, 11, 12,
            13, 14, 15, 16
        ]
    def create_output_data(self):
        return im2col_evaluate(
            self.create_input_data(),
            [1, 1, 4, 4],
            [self.kernel, self.kernel],
            [self.stride, self.stride],
            [self.pad, self.pad, self.pad, self.pad],
            [self.dilation, self.dilation]
        )

    def create_test_data(self):
        return TestData(
            name=self.name,
            input_data=self.create_input_data(),
            input_shape=[1, 1, 4, 4],
            output_data=self.create_output_data()
        )

class Im2ColTest1(_BaseIm2Col3x3):
    def __init__(self, base_path):
        super().__init__(base_path, kernel=3, stride=1, pad=0, dilation=1)
    
class Im2ColTest2(_BaseIm2Col3x3):
    def __init__(self, base_path):
        super().__init__(base_path, kernel=3, stride=1, pad=1, dilation=1)
    
class Im2ColTest3(_BaseIm2Col4x4):
    def __init__(self, base_path):
        super().__init__(base_path, kernel=3, stride=2, pad=1, dilation=1)

class Im2ColTest4(_BaseIm2Col4x4):
    def __init__(self, base_path):
        super().__init__(base_path, kernel=3, stride=1, pad=1, dilation=2)