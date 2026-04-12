extends Node

var _texture: Texture2DRD
var _engine: MLInferenceEngine

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	_engine = MLInferenceEngine.new()
	_engine.init()
	var model_id = _engine.register_model("ml/tests/test_conv2d_stride_2.onnx")
	_engine.print_model(model_id)
	var texture_rid = _create_texture(3, 3)
	_texture = Texture2DRD.new()
	_texture.texture_rd_rid = texture_rid
	
	var descriptor = InferenceDescriptor.new()
	descriptor.add_texture_input("input", _texture)
	descriptor.add_float_array_output("output", "result")
	var task = _engine.queue_request(model_id, descriptor)
	task.completed.connect(_on_task_completed.bind(task, _engine))
	
func _on_task_completed(task: InferenceTask, engine: MLInferenceEngine):
	var output = engine.get_task_output(task, "result")
	print("output: ", output)
	print("size: ", output.size())
	engine.destroy_task(task)

func _create_texture(w: int, h: int) -> RID:
	var rd = RenderingServer.get_rendering_device()
	var format = RDTextureFormat.new()
	format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	format.width = w
	format.height = h
	format.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	format.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT \
		| RenderingDevice.TEXTURE_USAGE_STORAGE_BIT \
		| RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT
	
	var view = RDTextureView.new()
	var data = PackedByteArray()
	data.resize(w * h * 4)
	data.fill(255)
	var texture_rid = rd.texture_create(format, view, [data])
	return texture_rid
