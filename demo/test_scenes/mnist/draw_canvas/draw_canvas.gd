extends SubViewport

signal changed

@export var lines_parent: Control
@export var line_thickness_scale = 0.05

var _drawing: bool = false
var _current_line: Line2D

var line_width:
	get:
		return line_thickness_scale * size.x

func clear():
	while lines_parent.get_child_count():
		lines_parent.remove_child(lines_parent.get_child(0))

func _input(event: InputEvent) -> void:
	if event.is_action_pressed("click"):
		_drawing = true
	if event.is_action_released("click"):
		_drawing = false
		_current_line = null

	if event is InputEventMouse and _drawing:
		_add(event.position)


func _add(pos: Vector2):
	
	if _current_line == null:
		_current_line = Line2D.new()
		_current_line.width = line_width
		lines_parent.add_child(_current_line)
	
	_current_line.add_point(pos)
	changed.emit()
