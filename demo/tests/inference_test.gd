extends Node

class Test:
	var onnx_path: String
	var input: PackedFloat32Array
	var expected_output: PackedFloat32Array
	
	func _init(path, test_input, test_output) -> void:
		self.onnx_path = path
		self.input = test_input
		self.expected_output = test_output

func _ready() -> void:
	var tests = _setup_tests()
	_run_tests(tests)

func _setup_tests() -> Dictionary:
	var tests: Dictionary = {}
	
	# Weights is identiy. Bias is zeros
	tests["gemm_zeros_test"] = Test.new(
		"ml/tests/test_gemm.onnx",
		[0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0]
	)
	
	# Weights is identiy. Bias is zeros
	tests["gemm_identity_test"] = Test.new(
		"ml/tests/test_gemm.onnx",
		[1.0, 2.0, 3.0],
		[1.0, 2.0, 3.0]
	)
	
	# Bias is ones. Adds ones to input
	tests["gemm_bias_ones"] = Test.new(
		"ml/tests/test_gemm_bias_ones.onnx",
		[0.0, 0.0, 0.0],
		[1.0, 1.0, 1.0]
	)

	# Bias is ones. Adds ones to input
	tests["relu"] = Test.new(
		"ml/tests/test_relu.onnx",
		[-1.0, 0.0, 1.0],
		[ 0.0, 0.0, 1.0]
	)

	# Bias is ones. Adds ones to input
	var sigmoid_input = [-1.0, 0.0, 1.0]
	var sigmoid = func(x): return 1.0 / (1.0 + exp(-x))
	tests["sigmoid"] = Test.new(
		"ml/tests/test_sigmoid.onnx",
		sigmoid_input,
		sigmoid_input.map(sigmoid)
	)
	
	return tests
	
func _run_tests(tests: Dictionary):
	var engine = MLInferenceEngine.new()
	
	for test_name in tests:
		
		var test: Test = tests[test_name]
		
		if not engine.load(test.onnx_path):
			push_error("Test %s failed to load" % [test_name])
			
		engine.run(test.input)
		var output = engine.get_output_data("output")
		assert_almost_equals(test_name, test.expected_output, output)
		
		engine.unload()
		await get_tree().create_timer(0.1).timeout
	
func assert_almost_equals(test_name, a: PackedFloat32Array, b: PackedFloat32Array, epsilon = 0.0001):
	if a.size() != b.size():
		push_error("Size mismatch")
		return
	for i in range(a.size()):
		if abs(a[i] - b[i]) > epsilon:
			push_error("Test %s failed at index %d. Expected %f, got %f" % [test_name, i, a[i], b[i]])
			return
	print("Success: %s. Got: %s" % [test_name, b])
