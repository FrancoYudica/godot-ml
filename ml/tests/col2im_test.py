import onnx
from onnx import helper, TensorProto
import numpy as np
import onnxruntime as ort

from onnx import helper, TensorProto
import onnx

def create_col2im_model():
    # 1. Define symbolic input: [Batch, (Channels * kH * kW), Total_Blocks]
    # 'M' and 'N' allow the input data size to vary, but the internal
    # math will be governed by the initializers below.
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 'M', 'N'])
    
    # 2. Define the output info
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 'H', 'W'])

    # 3. Create initializers (Internal constants)
    # This keeps the model input list clean (only one input needed)
    image_shape = helper.make_tensor('image_shape', TensorProto.INT64, [2], [2, 2])
    block_shape = helper.make_tensor('block_shape', TensorProto.INT64, [2], [3, 3])

    # 4. Create the Node
    node = helper.make_node(
        'Col2Im',
        inputs=['input', 'image_shape', 'block_shape'],
        outputs=['output'],
        dilations=[1, 1],
        pads=[1, 1, 1, 1],
        strides=[1, 1]
    )

    # 5. Build the Graph
    # We only put 'input' in the inputs list. 
    # image_shape and block_shape are handled via the initializer.
    graph = helper.make_graph(
        [node], 
        'Col2ImModelSimple', 
        [input_info], 
        [output_info], 
        [image_shape, block_shape]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    onnx.save(model, 'test_col2im.onnx')
    print("Saved test_col2im.onnx")

create_col2im_model()


session = ort.InferenceSession('test_col2im.onnx')

# Test Case: 2x2 output with 3x3 kernel

input_data = np.array([
    [0, 0, 0, 0, 1, 2, 0, 3, 4], # Represents center pixel (0,0)
    [0, 0, 0, 1, 2, 0, 3, 4, 0], # Represents center pixel (0,1)
    [0, 1, 2, 0, 3, 4, 0, 0, 0], # Represents center pixel (1,0)
    [1, 2, 0, 3, 4, 0, 0, 0, 0]  # Represents center pixel (1,1)
], dtype=np.float32).T.reshape(1, 9, 4)

# Only one input is required in the dictionary
outputs = session.run(None, {'input': input_data})

print("Output shape:", outputs[0].shape) # [1, 1, 2, 2]
print("Output data:")
print(outputs[0])