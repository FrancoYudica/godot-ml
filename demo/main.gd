extends Control

@export var sub_viewport: SubViewport

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var inference = MLInferenceEngine.new()
	var success = inference.load("ml/color_mlp.onnx")
	if not success:
		print("Failure")
	
	var data := PackedFloat32Array()
	
	data.push_back(0)
	data.push_back(0)
	data.push_back(0)
	data.push_back(0)
	inference.run(data)
	
	await get_tree().create_timer(1).timeout
	
	print(inference.get_output_data("net.2.bias"))
