extends Node2D


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var inference = MLInferenceEngine.new()
	inference.setup("ml/color_mlp.onnx")
