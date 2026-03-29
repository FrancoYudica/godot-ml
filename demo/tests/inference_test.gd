extends Node

@export var run: bool = true

class Test:
	var name: String
	var onnx_path: String
	var input: PackedFloat32Array
	var expected_output: PackedFloat32Array
	
	func _init(_name, path, test_input, test_output) -> void:
		self.name = _name
		self.onnx_path = path
		self.input = PackedFloat32Array(test_input)
		self.expected_output = PackedFloat32Array(test_output)

var engine = MLInferenceEngine.new()
var model_cache: Dictionary = {} # Path -> RID

func _ready() -> void:
	if not run:
		return
		
	# Initialize Engine
	engine.init()
	
	# Setup and Run
	var tests = _setup_tests()
	for i in range(1):
		_run_suite(tests)
	
	engine.destroy()

func _setup_tests() -> Array[Test]:
	var list: Array[Test] = []
	
	list.append(Test.new("Gemm Identity", "ml/tests/test_gemm.onnx", 
		[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
		
	list.append(Test.new("ReLU Basic", "ml/tests/test_relu.onnx", 
		[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]))
	
	var sig_in = [-1.0, 0.0, 1.0]
	var sig_out = sig_in.map(func(x): return 1.0 / (1.0 + exp(-x)))
	list.append(Test.new("Sigmoid", "ml/tests/test_sigmoid.onnx", sig_in, sig_out))
	
	return list

func _run_suite(tests: Array[Test]):
	for test in tests:
		# Avoid re-registering if we use the same model twice
		var model_id = model_cache.get(test.onnx_path, 0)
		if model_id == 0:
			model_id = engine.register_model(test.onnx_path)
			model_cache[test.onnx_path] = model_id
			
		if model_id == 0:
			push_error("FAILED: Could not register model for %s" % test.name)
			continue
			
		# Run Inference
		var task = engine.run_async(model_id)
		engine.add_float_array_input(task, "input", test.input, [1, 3])
		engine.add_float_array_output(task, "output")
		# Connect with binds so the callback knows wich test just finished
		task.completed.connect(_on_test_completed.bind(test, task))

func _on_test_completed(test: Test, task: InferenceTask):
	var result = engine.pop_task_output(task, "output")
	assert_almost_equals(test.name, test.expected_output, result)


func assert_almost_equals(test_name, a: PackedFloat32Array, b: PackedFloat32Array, epsilon = 0.0001):
	if a.size() != b.size():
		push_error("Size mismatch")
		return
	for i in range(a.size()):
		if abs(a[i] - b[i]) > epsilon:
			push_error("Test %s failed at index %d. Expected %f, got %f" % [test_name, i, a[i], b[i]])
			return
	print("Success: %s. Got: %s" % [test_name, b])
