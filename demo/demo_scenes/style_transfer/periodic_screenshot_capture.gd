extends Node

@export var sub_viewport: SubViewport
@export var delay_between_frames: int = 300
@export var target_count: int = -1 # No limit

@export var images_root_path: String
var _capture_counter: int = 0

func _process(_delta: float) -> void:
	
	if _capture_counter == target_count:
		return
	
	var time = Time.get_ticks_msec()
	if _capture_counter < time / delay_between_frames:
		_capture()


func _capture():
	var filepath = "%s\\%s.png" % [images_root_path, _capture_counter]
	_save_image(filepath, sub_viewport)
	_capture_counter += 1
	
func _save_image(filepath: String, sub_viewport: SubViewport):
	var image = sub_viewport.get_texture().get_image()
	image.save_png(filepath)
