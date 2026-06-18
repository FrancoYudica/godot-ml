extends Control

@export var runner: Node
@export var draw_canvas: Node
@export var result_label: Label

var queued_dispatch: bool = false

func _process(_delta: float) -> void:
	if queued_dispatch:
		runner.dispatch()

func _on_draw_canvas_changed() -> void:
	queued_dispatch = true

func _on_clear_button_pressed() -> void:
	draw_canvas.clear()


func _on_mnist_runner_predicted_number(value: int) -> void:
	result_label.text = str(value)
