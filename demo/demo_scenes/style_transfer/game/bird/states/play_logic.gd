extends BirdStateLogic

func on_state_entered():
	pass
	
func on_state_exited():
	pass

func on_state_unhandled_input(input: InputEvent):
	if input.is_action_pressed("click"):
		bird.velocity.y = -50
