import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort

def create_conv_transpose_identity():
    # 1. Define Shapes
    # Input: [Batch, Channels, Height, Width] -> [1, 1, 2, 2]
    # Weights: [In_Channels, Out_Channels/Group, kH, kW] -> [1, 1, 3, 3]
    input_shape = [1, 1, 2, 2]
    weight_shape = [1, 1, 3, 3]
    output_shape = [1, 1, 2, 2] # Result of s=1, p=1, k=3 on a 2x2

    # 2. Create Identity Weights
    # For a Transposed Conv, an "Identity" weight that preserves values 
    # usually has a 1 in the center and 0 elsewhere.
    weights = np.ones(weight_shape, dtype=np.float32)
    bias = np.zeros([1], dtype=np.float32)

    # 3. Define Model Inputs/Outputs
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

    # 4. Create Initializers (Weights and Bias)
    weights_initializer = helper.make_tensor('weights', TensorProto.FLOAT, weight_shape, weights.flatten())
    bias_initializer = helper.make_tensor('bias', TensorProto.FLOAT, [1], bias.flatten())

    # 5. Create the ConvTranspose Node
    # Note: ConvTranspose uses attributes for kernel, stride, and padding
    node = helper.make_node(
        'ConvTranspose',
        inputs=['input', 'weights', 'bias'],
        outputs=['output'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1], # [pad_top, pad_left, pad_bottom, pad_right]
        group=1,
        dilations=[1, 1]
    )

    # 6. Build the Graph
    graph = helper.make_graph(
        [node],
        'ConvTransposeTest',
        [input_info],
        [output_info],
        [weights_initializer, bias_initializer]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    onnx.save(model, 'test_conv_transpose.onnx')
    print("Saved test_conv_transpose.onnx")

# --- EXECUTION AND VERIFICATION ---
create_conv_transpose_identity()

# Test with your [1, 2, 3, 4] input
session = ort.InferenceSession('test_conv_transpose.onnx')
test_input = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)

res = session.run(None, {'input': test_input})

print("\nInput Matrix (2x2):")
print(test_input[0, 0])
print("\nConvTranspose Output (2x2) with k=3, s=1, p=1, all-ones weights:")
print(res[0][0, 0])