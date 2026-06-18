class_name StateLogic extends Node

@export var state: StateChartState

func _ready() -> void:
	state.state_entered.connect(on_state_entered)
	state.state_exited.connect(on_state_exited)
	state.state_processing.connect(on_state_processing)
	state.state_unhandled_input.connect(on_state_unhandled_input)

func on_state_processing(_delta: float):
	pass

func on_state_unhandled_input(_input: InputEvent):
	pass

func on_state_entered():
	pass
	
func on_state_exited():
	pass
