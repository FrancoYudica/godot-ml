extends SubViewport

signal changed

@export var point: PackedScene
@export var points_parent: Control
@export var point_scale = 0.1

var _drawing: bool = false
var _last_position: Vector2 = Vector2(-1, -1)

var point_size:
	get:
		return point_scale * size.x

func clear():
	while points_parent.get_child_count():
		points_parent.remove_child(points_parent.get_child(0))

func _input(event: InputEvent) -> void:
	if event.is_action_pressed("click"):
		_drawing = true
	if event.is_action_released("click"):
		_drawing = false
		_last_position = Vector2(-1, -1)

	if event is InputEventMouse and _drawing:
		_moved(event.position)
		_last_position = event.position

func _moved(pos: Vector2):
	
	if _last_position.x == -1 and _last_position.y == -1:
		_add(pos)
		return
	
	var distance = (pos - _last_position).length()
	var points = int(distance + point_size - 1) / int(point_size)
	
	var start = _last_position
	var end = pos
	for i in range(points):
		var t = float(i + 1) / points
		var position = lerp(start, end, t)
		_add(position)
	

func _add(pos: Vector2):
	var p = point.instantiate()
	p.position = pos
	p.size.x = point_size
	p.size.y = point_size
	points_parent.add_child(p)
	changed.emit()
