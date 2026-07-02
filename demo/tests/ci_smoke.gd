extends Node

func _ready() -> void:
	print("CI smoke test: Godot booted and ran a scene successfully.")
	get_tree().quit(0)
