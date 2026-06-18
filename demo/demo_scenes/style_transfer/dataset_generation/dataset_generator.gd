extends Control

@export var low_res_subviewport: SubViewport
@export var high_res_subviewport: SubViewport
@export var delay_between_frames: int = 300

@export var images_root_path: String
var _capture_counter: int = 0

func _process(_delta: float) -> void:
	var time = Time.get_ticks_msec()
	if _capture_counter < time / delay_between_frames:
		_capture()

func _unhandled_input(event: InputEvent) -> void:
	high_res_subviewport.push_input(event)

func _capture():
	var root_path = images_root_path
	var low_res_filepath = "%s\\low_res\\%s.png" % [root_path, _capture_counter]
	var high_res_filepath = "%s\\high_res\\%s.png" % [root_path, _capture_counter]
	_save_image(low_res_filepath, low_res_subviewport)
	_save_image(high_res_filepath, high_res_subviewport)
	
	_capture_counter += 1
	print("Saved images")
	
func _save_image(filepath: String, sub_viewport: SubViewport):
	var image = sub_viewport.get_texture().get_image()
	image.save_png(filepath)
