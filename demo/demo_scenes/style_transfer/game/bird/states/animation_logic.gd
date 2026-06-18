extends BirdStateLogic

@export var sprite: Sprite2D

@export var max_speed: float = 70.0  
@export var smooth_speed: float = 15.0

@export var fall_stretch_y: float = 1.4  # Taller
@export var fall_squash_x: float = 0.6   # Thinner

@export var jump_squash_y: float = 0.6   # Shorter (Compress vertically)
@export var jump_stretch_x: float = 1.4  # Wider (Expand horizontally)

func on_state_processing(delta: float):
	var speed_y = abs(bird.velocity.y)
	var stretch_weight = clampf(speed_y / max_speed, 0.0, 1.0)
	
	var target_scale_x = 1.0
	var target_scale_y = 1.0
	
	# Check the direction of the velocity
	if bird.velocity.y < 0:
		# JUMPING: Compress Y, Expand X
		target_scale_x = lerp(1.0, jump_stretch_x, stretch_weight)
		target_scale_y = lerp(1.0, jump_squash_y, stretch_weight)
		
	elif bird.velocity.y > 0:
		# FALLING: Expand Y, Compress X
		target_scale_x = lerp(1.0, fall_squash_x, stretch_weight)
		target_scale_y = lerp(1.0, fall_stretch_y, stretch_weight)
		
	var target_scale = Vector2(target_scale_x, target_scale_y)
	
	# Smoothly apply the target scale
	sprite.scale = sprite.scale.lerp(target_scale, smooth_speed * delta)
