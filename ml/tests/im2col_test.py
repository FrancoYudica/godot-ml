import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort

def create_im2col_model():
    input_info  = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1, 'C', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)

    node = helper.make_node(
        'Im2Col',
        inputs=['input'],
        outputs=['output'],
        dilations=[1, 1],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1]
    )

    graph = helper.make_graph([node], 'Im2ColModel', [input_info], [output_info], [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    onnx.save(model, 'test_im2col.onnx')
    print("Saved test_im2col.onnx")

create_im2col_model()



def im2col_reference(input_data, kernel_size, pad, stride):
    N, C, H, W = input_data.shape
    kH = kW = kernel_size
    out_h = (H + 2 * pad - kH) // stride + 1
    out_w = (W + 2 * pad - kW) // stride + 1

    # Pad input
    padded = np.pad(input_data, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')

    cols = []
    for y in range(out_h):
        for x in range(out_w):
            patch = padded[:, :, y*stride:y*stride+kH, x*stride:x*stride+kW]
            cols.append(patch.flatten())

    return np.array(cols)  # [out_h*out_w, C*kH*kW]

input_data = np.array([[[[1,2,3],[4,5,6],[7,8,9]]]], dtype=np.float32)  # [1,1,3,3]
ref = im2col_reference(input_data, kernel_size=3, pad=1, stride=1)
print(ref.flatten().tolist())