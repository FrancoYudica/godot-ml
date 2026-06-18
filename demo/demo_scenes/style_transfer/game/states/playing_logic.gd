extends StateLogic

@export var frame: Node2D
@export var speed: float = 1.0

func on_state_processing(_delta: float):
	frame.position.x += _delta * speed
