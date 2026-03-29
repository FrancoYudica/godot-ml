extends Node

@export var viewport: SubViewport  # Better than just a Texture2D
@export var texture_rect: TextureRect

var engine: MLInferenceEngine
var model_id: int = 0
var is_processing_model: bool = false
var result_texture: ImageTexture

func _ready() -> void:
	engine = MLInferenceEngine.new()
	engine.init()
	
	# Pre-register the model once
	model_id = engine.register_model("ml/sepia_mlp.onnx")
	
	# Initialize the result texture once to avoid allocation stutters
	result_texture = ImageTexture.new()

func _process(_delta: float) -> void:
	if not viewport or model_id == 0:
		return
		
	# THREAD SAFETY: Don't start a new task if the GPU is still working on the last one
	if not is_processing_model:
		_dispatch_inference()

func _dispatch_inference() -> void:
	var t = Time.get_ticks_usec()

	var task = engine.run_async(model_id)
	var tex = viewport.get_texture()
	engine.add_texture_input(task, "pixels", tex)
	engine.add_float_array_output(task, "result", "result_float_array")
	task.completed.connect(_on_inference_completed.bind(task, tex.get_size()), CONNECT_ONE_SHOT)
	var elapsed = Time.get_ticks_usec() - t
	print("Dispatch took: %s ms" % [elapsed / 1000.0])
	
func _on_inference_completed(task: InferenceTask, src_size: Vector2i) -> void:
	
	var t = Time.get_ticks_usec()
	# 1. Grab the output from the engine
	var result_buffer = engine.get_task_output(task, "result_float_array")
	engine.destroy_task(task)
	# 2. Reconstruct the image
	var img = Image.create_from_data(
		src_size.x, 
		src_size.y, 
		false, 
		Image.Format.FORMAT_RGBF, 
		result_buffer.to_byte_array()
	)
	
	# 3. Update existing texture instead of creating a new one
	result_texture.set_image(img)
	texture_rect.texture = result_texture
	var elapsed = Time.get_ticks_usec() - t
	print("Getting output took: %s ms" % [elapsed / 1000.0])
	# 4. Ready for the next frame
	is_processing_model = false
