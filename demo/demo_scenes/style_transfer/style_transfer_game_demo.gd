extends Control

@export var high_res_sub_viewport: SubViewport

func _unhandled_input(event: InputEvent) -> void:
	high_res_sub_viewport.push_input(event)
