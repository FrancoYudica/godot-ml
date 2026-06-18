extends BirdStateLogic


func on_state_processing(delta: float):
	bird.velocity.y += delta * 100.0
	bird.position += bird.velocity * delta
