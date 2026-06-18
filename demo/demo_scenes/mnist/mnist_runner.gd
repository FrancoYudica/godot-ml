extends Node

signal predicted_number(value: int)

@export var input_texture_viewport: SubViewport
@export var onnx_model: ONNXResource

var engine: MLInferenceEngine
var model_id: int = 0

func _ready() -> void:
	engine = MLInferenceEngine.new()
	engine.init()
	model_id = engine.register_model(onnx_model)
	engine.print_model(model_id)

func dispatch():
	_dispatch_inference()


func _dispatch_inference() -> void:
	var descriptor = InferenceDescriptor.new()
	var tex = input_texture_viewport.get_texture()
	var input_texture = input_texture_viewport.get_texture()
	
	descriptor.add_texture_input(
		"x", 
		tex, 
		InferenceDescriptor.TEXTURE_LOAD_RED, 
		input_texture.get_width(), 
		input_texture.get_height()
	)
	descriptor.add_float_array_output("linear", "output")
	var task = engine.queue_request(model_id, descriptor)
	if task == null:
		return
		
	task.completed.connect(_on_inference_completed.bind(task), CONNECT_ONE_SHOT)

func _on_inference_completed(task: InferenceTask) -> void:
	var result = engine.get_task_output(task, "output")
	
	var max_i = 0
	for i in range(1, 10):
		if result[max_i] < result[i]:
			max_i = i
	
	engine.destroy_task(task)

	predicted_number.emit(max_i)
