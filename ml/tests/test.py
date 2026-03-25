import onnx
from onnx import helper, TensorProto
import numpy as np

def create_gemm_model():
    # 'N' is the symbolic dimension for arbitrary input rows
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['N', 3])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['N', 3])

    # 3x3 Identity Matrix
    weights = np.eye(3).astype(np.float32)
    bias = np.zeros(3).astype(np.float32)

    w_init = helper.make_tensor('W', TensorProto.FLOAT, [3, 3], weights.flatten())
    b_init = helper.make_tensor('B', TensorProto.FLOAT, [3], bias.flatten())

    node = helper.make_node('Gemm', ['input', 'W', 'B'], ['output'], alpha=1.0, beta=1.0, transB=1)

    graph = helper.make_graph([node], 'GemmModel', [input_info], [output_info], [w_init, b_init])
    onnx.save(helper.make_model(graph), 'test_gemm.onnx')


def create_relu_model():
    # We use 'N' and 'K' to show it can handle any 2D shape
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['N', 'K'])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['N', 'K'])
    node = helper.make_node('Relu', ['input'], ['output'])
    graph = helper.make_graph([node], 'ReluModel', [input_info], [output_info], [])
    onnx.save(helper.make_model(graph), 'test_relu.onnx')

def create_sigmoid_model():
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['N', 'K'])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['N', 'K'])
    node = helper.make_node('Sigmoid', ['input'], ['output'])
    graph = helper.make_graph([node], 'SigmoidModel', [input_info], [output_info], [])
    onnx.save(helper.make_model(graph), 'test_sigmoid.onnx')

create_sigmoid_model()
create_relu_model() 
create_gemm_model()