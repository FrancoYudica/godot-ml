extends Node

@export var gaps_scene: PackedScene
@export var gaps_parent: Node2D
@export var gaps_max_space: int =  35
@export var gaps_min_space: int =  20
@export var horizontal_space: int =  30
@export var space_curve: Curve
@export var camera: Camera2D
@export var start_spawn_x: int = 60

var gaps: Array[FlappyGap] = []
var _pooled_gaps: Array[FlappyGap] = []
var viewport_size_x = 128.0
var half_width:
	get:
		return viewport_size_x * 0.5

var visibility_padding = 20

func _is_visible(x):
	var lower_bound = x > camera.position.x - half_width - visibility_padding
	var upper_bound = x < camera.position.x + half_width + visibility_padding
	return lower_bound and upper_bound

func _process(delta: float) -> void:
	
	if gaps.size() == 0:
		_spawn(start_spawn_x)
	
	var last_x = gaps.back().position.x
	var next_x = last_x + horizontal_space
	
	# Will it be visible when the camera moves 10 pixels?
	if _is_visible(next_x - 10): 
		_spawn(next_x)
	
	var remove_to_index = -1
	
	for i in range(gaps.size()):
		var gap = gaps[i]
		if not _is_visible(gap.position.x):
			remove_to_index = i
			
		else:
			break
	
	while remove_to_index >= 0:
		_free_gap(gaps[0])
		remove_to_index -= 1
		
func _spawn(x: float):
	var gap = _spawn_gap()
	gap.position.x = x
	gap.gap = lerpf(gaps_max_space, gaps_min_space, space_curve.sample(0.0))
	gap.position.y = randf_range(-7.5, 7.5)

func _free_gap(gap: FlappyGap):
	gaps.erase(gap)
	_pooled_gaps.append(gap)
	gap.process_mode = Node.PROCESS_MODE_DISABLED
	
func _spawn_gap():
	
	var gap: FlappyGap = null
	
	if _pooled_gaps.size() == 0:
		gap = gaps_scene.instantiate()
		gaps_parent.add_child(gap)
		print("Insantiated gap")
	else:
		gap = _pooled_gaps.pop_back()
		gap.process_mode = Node.PROCESS_MODE_ALWAYS
		print("Got pooled gap")
		
	gaps.append(gap)
	return gap
