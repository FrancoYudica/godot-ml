import onnx
from onnx import helper, TensorProto
import numpy as np
import onnxruntime as ort

def create_conv_transpose_4x4():
    # 1. Define Shapes
    # Input: [Batch, Channels, Height, Width]
    in_channels = 4
    out_channels = 4
    k_size = 3
    input_shape = [1, in_channels, 2, 2]
    # ConvTranspose Weight Shape: [In_C, Out_C/Group, kH, kW]
    weight_shape = [in_channels, out_channels, k_size, k_size]
    output_shape = [1, out_channels, 2, 2] 

    # 2. Create Deterministic Weights (All 1.0) and Bias (All 0.0)
    w_values = np.ones(weight_shape, dtype=np.float32)
    b_values = np.zeros([out_channels], dtype=np.float32)

    # 3. Define Model Inputs/Outputs
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

    # 4. Create Initializers
    weights_init = helper.make_tensor('weights', TensorProto.FLOAT, weight_shape, w_values.flatten())
    bias_init = helper.make_tensor('bias', TensorProto.FLOAT, [out_channels], b_values.flatten())

    # 5. Create the ConvTranspose Node
    node = helper.make_node(
        'ConvTranspose',
        inputs=['input', 'weights', 'bias'],
        outputs=['output'],
        kernel_shape=[k_size, k_size],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        name='conv_transpose_layer'
    )

    # 6. Build Graph and Model
    graph = helper.make_graph(
        [node],
        'ConvTranspose4x4Test',
        [input_info],
        [output_info],
        [weights_init, bias_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    onnx.save(model, 'test_conv_transpose_4x4.onnx')
    print("Saved test_conv_transpose_4x4.onnx")

create_conv_transpose_4x4()

# --- VERIFICATION ---
session = ort.InferenceSession('test_conv_transpose_4x4.onnx')
# Create a simple input where each channel i has the value i+1
# e.g., Channel 0 is all 1s, Channel 1 is all 2s, etc.
test_input = np.zeros((1, 4, 2, 2), dtype=np.float32)
for i in range(4):
    test_input[0, i, :, :] = i + 1

res = session.run(None, {'input': test_input})

print("\nChannel 0 of Input (All 1.0s):")
print(test_input)
print("\nOutput Channel 0 (Deterministic Sum):")
print(res)