extends Node

@export var viewport: SubViewport  # Better than just a Texture2D
@export var texture_rect: TextureRect

var engine: MLInferenceEngine
var model_id: int = 0
var is_processing: bool = false
var result_texture: ImageTexture

func _ready() -> void:
	engine = MLInferenceEngine.new()
	engine.init()
	
	# Pre-register the model once
	model_id = engine.register_model("ml/color_mlp.onnx")
	
	# Initialize the result texture once to avoid allocation stutters
	result_texture = ImageTexture.new()

func _process(_delta: float) -> void:
	if not viewport or model_id == 0:
		return
		
	# THREAD SAFETY: Don't start a new task if the GPU is still working on the last one
	if not is_processing:
		_dispatch_inference()

func _dispatch_inference() -> void:
	var tex = viewport.get_texture()
	var img = tex.get_image() # This is the CPU bottleneck
	
	# Optimization: If the viewport is already RGBF, this is much faster
	if img.get_format() != Image.Format.FORMAT_RGBF:
		img.convert(Image.Format.FORMAT_RGBF)
	
	var src_size = img.get_size()
	var input_data = img.get_data().to_float32_array()
	
	is_processing = true
	
	var task = engine.run_async(model_id, input_data)
	task.completed.connect(_on_inference_completed.bind(task, src_size), CONNECT_ONE_SHOT)

func _on_inference_completed(task: InferenceTask, src_size: Vector2i) -> void:
	# 1. Grab the output from the engine
	var result_buffer = engine.pop_task_output(task, "result")
	
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
	
	# 4. Ready for the next frame
	is_processing = false
