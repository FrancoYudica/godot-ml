extends Node

@export var run: bool = true

class Test:
	var name: String
	var onnx_path: String
	var setup_callable: Callable
	var pop_output_callable: Callable
	var expected_output: PackedFloat32Array
	
	func _init(_name, path, setup_callable, pop_output_callable, test_output) -> void:
		self.name = _name
		self.onnx_path = path
		self.setup_callable = setup_callable
		self.pop_output_callable = pop_output_callable
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

	list.append(
		Test.new(
			"Gemm Identity", 
			"ml/tests/test_gemm.onnx", 
			_setup_float_array_test.bind([1.0, 2.0, 3.0], [1, 3]),
			_pop_float_array_result,
			[1.0, 2.0, 3.0]
			)
	)
	list.append(
		Test.new(
			"ReLU Basic",
			"ml/tests/test_relu.onnx",
			_setup_float_array_test.bind([-1.0, 0.0, 1.0], [1, 3]),
			_pop_float_array_result,
			[0.0, 0.0, 1.0]
		)
	)
	
	var sig_in = [-1.0, 0.0, 1.0]
	
	list.append(
		Test.new(
			"Sigmoid",
			"ml/tests/test_sigmoid.onnx",
			_setup_float_array_test.bind(sig_in, [1, 3]),
			_pop_float_array_result,
			sig_in.map(func(x): return 1.0 / (1.0 + exp(-x)))
		)
	)
	
	var white_image_floats: PackedFloat32Array
	white_image_floats.resize(3 * 3 * 3) # 3x3 RGB image
	white_image_floats.fill(1.0)
	
	var expected_result: PackedFloat32Array
	expected_result.resize(32)
	expected_result.fill(12)
	
	list.append(
		Test.new(
			"Conv2D",
			"ml/tests/test_conv2d_stride_2.onnx",
			_setup_float_array_test.bind(
				white_image_floats, 
				[
					1, # Batches
					3, # Channels
					3, # Height
					3] # Width
				),
			_pop_float_array_result,
			expected_result
		)
	)
	
	var input_im = [
		1, 2, 3,
		4, 5, 6,
		7, 8, 9
	]
	
	list.append(
		Test.new(
			"Img2Col",
			"ml/tests/test_im2col.onnx",
			_setup_float_array_test.bind(
				input_im, 
				[
					1, # Batches
					1, # Channels
					3, # Height
					3
				] # Width
			),
			_pop_float_array_result,
			
			# Expected output
			[
				0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0, # Patch 0
				0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, # Patch 1
				0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, # Patch 2
				0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, # Patch 3
				1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, # Patch 4
				2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0, # Patch 5
				0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, # Patch 6
				4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, # Patch 7
				5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0  # Patch 8
			]
		)
	)
	
	list.append(
		Test.new(
			"Col2Im",
			"ml/tests/test_conv_transpose.onnx",
			_setup_float_array_test.bind(
				[
					1, 2, 
					3, 4
				], 
				[
					1, # Batches
					1, # Channels
					2, # Height
					2  # Width
				]
			),
			_pop_float_array_result,
			# Expected output
			[
				10, 10,
				10, 10
			]
		)
	)
	return list

func _setup_float_array_test(descriptor: InferenceDescriptor, data: PackedFloat32Array, shape: PackedFloat64Array):
	descriptor.add_float_array_input("input", data, shape)
	descriptor.add_float_array_output("output", "output_float_array")

func _pop_float_array_result(task: InferenceTask) -> PackedFloat32Array:
	var data = engine.get_task_output(task, "output_float_array")
	return PackedFloat32Array(data) # Ensure it returns the expected type for assertion

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
		var descriptor = InferenceDescriptor.new()
		test.setup_callable.call(descriptor)
		var task = engine.queue_request(model_id, descriptor)

		# Connect with binds so the callback knows wich test just finished
		task.completed.connect(_on_test_completed.bind(test, task))

func _on_test_completed(test: Test, task: InferenceTask):
	var result = test.pop_output_callable.call(task)
	engine.destroy_task(task)
	assert_almost_equals(test.name, test.expected_output, result)


func assert_almost_equals(test_name, a: PackedFloat32Array, b: PackedFloat32Array, epsilon = 0.0001):
	if a.size() != b.size():
		push_error("Size mismatch")
		print(a)
		print(b)
		return
	for i in range(a.size()):
		if abs(a[i] - b[i]) > epsilon:
			push_error("Test %s failed at index %d. Expected %f, got %f" % [test_name, i, a[i], b[i]])
			print(a)
			print(b)
			return
	print("Success: %s. Got: %s" % [test_name, b])
