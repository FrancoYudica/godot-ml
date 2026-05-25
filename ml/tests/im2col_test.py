import onnx
from onnx import helper, TensorProto
import numpy as np

def create_im2col_model():
    input_info  = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1, 'C', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)

    node = helper.make_node(
        'Im2Col',
        inputs=['input'],
        outputs=['output'],
        dilations=[2, 2], 
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1]
    )

    graph = helper.make_graph([node], 'Im2ColModel', [input_info], [output_info], [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    onnx.save(model, 'test_im2col.onnx')
    print("Saved test_im2col.onnx")

create_im2col_model()