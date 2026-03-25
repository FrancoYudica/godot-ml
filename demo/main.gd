extends Control

@export var sub_viewport: SubViewport

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass
	#var inference = MLInferenceEngine.new()
	#var success = inference.load("ml/test_gemm_ones.onnx")
	#if not success:
		#print("Failure")
	#
	#var data := PackedFloat32Array()
	#
	#for i in range(2):
		#data.push_back(1.0)
		#data.push_back(2.0)
		#
	#inference.run(data)
	#
	#print(inference.get_output_data("output"))
