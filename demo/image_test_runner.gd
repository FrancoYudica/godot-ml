extends Node

@export var viewport: SubViewport  # Better than just a Texture2D
@export var texture_rect: TextureRect

var engine: MLInferenceEngine
var model_id: int = 0
var result_texture: Texture2D

var image_size = 1024

func _ready() -> void:
	engine = MLInferenceEngine.new()
	engine.init()
	model_id = engine.register_model("ml/sobel_filter.onnx")
	engine.print_model(model_id)
	
	# Initialize the result texture
	var rd = RenderingServer.get_rendering_device()
	var format = RDTextureFormat.new()
	format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	format.width = image_size
	format.height = image_size
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
	if not viewport or model_id == 0:
		return
		
	_dispatch_inference()
	
	print("FPS: %s" % (1.0 / _delta))

func _dispatch_inference() -> void:
	var descriptor = InferenceDescriptor.new()
	var tex = viewport.get_texture()
	descriptor.add_texture_input("input", tex, image_size, image_size)
	descriptor.add_texture_output("output", result_texture)
	var task = engine.queue_request(model_id, descriptor)
	if task == null:
		return
		
	task.completed.connect(_on_inference_completed.bind(task), CONNECT_ONE_SHOT)

func _on_inference_completed(task: InferenceTask) -> void:
	engine.destroy_task(task)
