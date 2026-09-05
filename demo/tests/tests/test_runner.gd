extends Node

@export var run: bool = true
@export_file("*.json") var test_suite_path: String

var engine = MLInferenceEngine.new()
var model_cache: Dictionary = {}

func _ready() -> void:
	if not run: return
	engine.init()
	_run_json_suite(test_suite_path)
	engine.destroy()

func _run_json_suite(path: String):
	var json_string = FileAccess.get_file_as_string(path)
	var json = JSON.new()
	var error = json.parse(json_string)
	
	if error != OK:
		push_error("JSON Parse Error: %s at line %d" % [json.get_error_message(), json.get_error_line()])
		return

	var test_data_array = json.get_data()
	for data in test_data_array:
		_execute_test(data)

func _execute_test(data: Dictionary):
	# Register/Get Model
	var model_id = _get_or_register_model(data.name + ".onnx")
	if model_id == 0: return

	# Configure
	var descriptor = InferenceDescriptor.new()
	descriptor.add_float_array_input("input", PackedFloat32Array(data.input_data), PackedFloat64Array(data.input_shape))
	descriptor.add_float_array_output("output", "output_float_array")

	# Run
	var task = engine.queue_request(model_id, descriptor)
	task.completed.connect(_on_test_completed.bind(data, task))

func _get_or_register_model(path: String) -> int:
	if model_cache.has(path): return model_cache[path]
	var id = engine.register_model(load(path))
	if id != 0: model_cache[path] = id
	else: push_error("Failed to load: " + path)
	return id

func _on_test_completed(data: Dictionary, task: InferenceTask):
	var result = PackedFloat32Array(engine.get_task_output(task, "output_float_array"))
	var expected = PackedFloat32Array(data.expected_output)
	
	_assert_almost_equals(data.name, expected, result)
	engine.destroy_task(task)

func _assert_almost_equals(test_name: String, a: PackedFloat32Array, b: PackedFloat32Array, epsilon = 0.0001):
	if a.size() != b.size():
		push_error("FAIL: %s | Size mismatch (Expected %d, Got %d)" % [test_name, a.size(), b.size()])
		return
	for i in range(a.size()):
		if abs(a[i] - b[i]) > epsilon:
			push_error("FAIL: %s | Mismatch at %d (Expected %f, Got %f)" % [test_name, i, a[i], b[i]])
			return
	print("SUCCESS: %s" % test_name)
