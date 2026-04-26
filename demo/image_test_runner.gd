extends Node

@export var input_texture_viewport: SubViewport
@export var texture_rect: TextureRect
@export var onnx_model_path: String
@export var scale_factor: int

var engine: MLInferenceEngine
var model_id: int = 0
var result_texture: Texture2D

func _ready() -> void:
	engine = MLInferenceEngine.new()
	engine.init()
	model_id = engine.register_model(onnx_model_path)
	engine.print_model(model_id)
	var input_texture = input_texture_viewport.get_texture()
	var upscale_size = Vector2i(
		input_texture.get_width() * scale_factor,
		input_texture.get_height() * scale_factor
	)
	# Initialize the result texture
	var rd = RenderingServer.get_rendering_device()
	var format = RDTextureFormat.new()
	format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	format.width = upscale_size.x
	format.height = upscale_size.y
	format.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	format.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT \
		| RenderingDevice.TEXTURE_USAGE_STORAGE_BIT \
		| RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT
	
	var view = RDTextureView.new()
	var texture_rid = rd.texture_create(format, view)
	result_texture = Texture2DRD.new()
	result_texture.texture_rd_rid = texture_rid
	texture_rect.texture = result_texture

func _process(_delta: float) -> void:
	if not input_texture_viewport or model_id == 0:
		return
		
	_dispatch_inference()
	
	print("FPS: %s" % (1.0 / _delta))

func _dispatch_inference() -> void:
	var descriptor = InferenceDescriptor.new()
	var tex = input_texture_viewport.get_texture()
	var input_texture = input_texture_viewport.get_texture()
	descriptor.add_texture_input("input", tex, input_texture.get_width(), input_texture.get_height())
	descriptor.add_texture_output("output", result_texture)
	var task = engine.queue_request(model_id, descriptor)
	if task == null:
		return
		
	task.completed.connect(_on_inference_completed.bind(task), CONNECT_ONE_SHOT)

func _on_inference_completed(task: InferenceTask) -> void:
	engine.destroy_task(task)
