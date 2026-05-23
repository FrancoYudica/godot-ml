import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

def create_and_evaluate_maxpool(kernel_size=2, stride=2, pad=0):
    in_channels = 3
    filename = 'test_maxpool2d.onnx'
    
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', in_channels, 'height', 'width'])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', in_channels, 'out_height', 'out_width'])

    node = helper.make_node(
        'MaxPool',
        inputs=['input'],
        outputs=['output'],
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad], 
        name='maxpool_layer'
    )

    graph = helper.make_graph(
        nodes=[node],
        name='MaxPool2DModel',
        inputs=[input_info],
        outputs=[output_info],
        initializer=[]
    )

    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    
    onnx.save(model, filename)
    print(f"[*] Successfully created model: {filename}")
    
    # Evaluate the Model directly via ONNX Runtime
    # Generate deterministic input shape: (1, 3, 4, 4)
    np.random.seed(42)
    input_shape = (1, 3, 4, 4)
    input_data = np.arange(1, 1 + np.prod(input_shape), dtype=np.float32).reshape(input_shape)
    
    # Fire up the runtime engine session using the file we just saved
    session = ort.InferenceSession(filename, providers=['CPUExecutionProvider'])
    
    # ONNX Runtime expectations: feed a dictionary map matching input name strings
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    output_data = outputs[0]
    
    # -------------------------------------------------------------
    # 3. Print Results for Godot Addon Matching
    # -------------------------------------------------------------
    print("\n================== ONNX RUNTIME EVALUATION ==================")
    print(f"Input Shape:  {input_shape}")
    print(f"Output Shape: {output_data.shape}")
    print("\n---> [Input Channel 0 Grid]:")
    print(input_data)
    print("\n---> [ONNX Evaluated Expected Output Channel 0]:")
    print(output_data)
    print("=============================================================")

if __name__ == "__main__":
    # Feel free to change kernel, stride, or pad parameters to test edge cases!
    create_and_evaluate_maxpool(kernel_size=2, stride=2, pad=0)