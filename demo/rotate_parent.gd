extends Node

func _process(delta: float) -> void:
	get_parent().rotation += delta
