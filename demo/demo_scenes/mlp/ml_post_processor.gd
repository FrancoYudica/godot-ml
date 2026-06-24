extends Node

@export var input_texture_viewport: SubViewport
@export var texture_rect: TextureRect

var engine: MLInferenceEngine
var model_id: int = 0
var result_texture: Texture2D

func _ready() -> void:
	engine = MLInferenceEngine.new()
	engine.init()
	var input_texture = input_texture_viewport.get_texture()
	# Initialize the result texture
	var rd = RenderingServer.get_rendering_device()
	var format = RDTextureFormat.new()
	format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	format.width = input_texture.get_width()
	format.height = input_texture.get_height()
	format.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	format.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT \
		| RenderingDevice.TEXTURE_USAGE_STORAGE_BIT \
		| RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT \
		| RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	
	var view = RDTextureView.new()
	var texture_rid = rd.texture_create(format, view)
	rd.texture_clear(texture_rid, Color(0, 0, 0, 1), 0, 1, 0, 1)
	result_texture = Texture2DRD.new()
	result_texture.texture_rd_rid = texture_rid
	texture_rect.texture = result_texture
	RenderingServer.frame_pre_draw.connect(_pre_draw)

func setup_model(model: ONNXResource):
	
	if model_id != 0:
		engine.unload_model(model_id)
	
	model_id = engine.register_model(model)


func _pre_draw():
	if not input_texture_viewport or model_id == 0:
		return
		
	_dispatch_inference()

func _dispatch_inference() -> void:
	var descriptor = InferenceDescriptor.new()
	var tex = input_texture_viewport.get_texture()
	var input_texture = input_texture_viewport.get_texture()
	
	descriptor.add_texture_input(
		"pixels", 
		tex, 
		InferenceDescriptor.TEXTURE_LOAD_RGB,
		input_texture.get_width(), 
		input_texture.get_height()
	)
	descriptor.add_texture_output("result", result_texture)
	var task = engine.queue_request(model_id, descriptor)
	if task == null:
		return
		
	task.completed.connect(_on_inference_completed.bind(task), CONNECT_ONE_SHOT)

func _on_inference_completed(task: InferenceTask) -> void:
	# Destroy task - Output uploaded automatically to texture
	engine.destroy_task(task)
