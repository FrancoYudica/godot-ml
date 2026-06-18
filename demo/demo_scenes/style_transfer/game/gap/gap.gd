class_name FlappyGap extends Node2D

@export var top_pipe: Node2D
@export var bottom_pipe: Node2D

@export var gap: int:
	set(value):
		gap = value
		top_pipe.position.y = -gap * 0.5
		bottom_pipe.position.y = gap * 0.5
