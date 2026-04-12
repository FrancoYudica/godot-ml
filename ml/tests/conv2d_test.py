import onnx
from onnx import helper, TensorProto
import numpy as np

def create_conv2d_model(stride=1):
    # Model parameters
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    
    # 1. Define Input/Output (N, C, H, W)
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', in_channels, 'height', 'width'])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', out_channels, 'height', 'width'])

    # 2. Define Weights (All 1.0)
    # Shape: [out_channels, in_channels, kH, kW]
    w_shape = [out_channels, in_channels, kernel_size, kernel_size]
    # Filling with 1.0 makes verification easy: output = sum(input_window)
    w_values = np.ones(w_shape).astype(np.float32).flatten()
    
    weights = helper.make_tensor(
        name='W',
        data_type=TensorProto.FLOAT,
        dims=w_shape,
        vals=w_values
    )

    # 3. Define Biases (All 0.0)
    b_shape = [out_channels]
    b_values = np.zeros(b_shape).astype(np.float32).flatten()
    
    biases = helper.make_tensor(
        name='B',
        data_type=TensorProto.FLOAT,
        dims=b_shape,
        vals=b_values
    )

    # 4. Create the Conv Node
    # pads=[1, 1, 1, 1] ensures output H/W == input H/W
    node = helper.make_node(
        'Conv',
        inputs=['input', 'W', 'B'],
        outputs=['output'],
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=[1, 1, 1, 1], 
        name='conv_layer'
    )

    # 5. Build the Graph
    graph = helper.make_graph(
        nodes=[node],
        name='Conv2DModel',
        inputs=[input_info],
        outputs=[output_info],
        initializer=[weights, biases]
    )

    model = helper.make_model(graph, producer_name='godot-ml-engine')
    
    # Set opset version (13+ is standard for modern features)
    model.opset_import[0].version = 13
    
    onnx.save(model, 'test_conv2d.onnx')
    print("Successfully created deterministic test_conv2d.onnx")
    print(f"Expected output for white texture (1.0) input: {float(in_channels * kernel_size * kernel_size)}")

create_conv2d_model(stride=2)