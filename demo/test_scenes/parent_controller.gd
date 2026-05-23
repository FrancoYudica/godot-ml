extends Node

@export var speed: float = 10
@export var rotation_speed: float = 3

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	var parent = get_parent()
	parent.position.x += delta * Input.get_axis("left", "right") * speed
	parent.position.y -= delta * Input.get_axis("down", "up") * speed
	parent.rotation += delta * (1.0 if Input.is_action_pressed("rotate") else 0.0) * rotation_speed
