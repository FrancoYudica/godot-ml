extends Node

var _t = 0.0

func _process(delta: float) -> void:
	_t += delta
	get_parent().position = Vector2(
		cos(_t),
		sin(_t)
	) * 4
