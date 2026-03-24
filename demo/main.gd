extends Node2D


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var inference = MLInferenceEngine.new()
	var success = inference.load("ml/color_mlp.onnx")
	if not success:
		print("Failure")
