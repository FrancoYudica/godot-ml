extends Control

@export var save_file_dialog: FileDialog
@export var sub_viewport: SubViewport
@export var capture: bool = false:
	set(value):
		if value:
			request_capture()
		capture = false

var _pending_image: Image

func _ready() -> void:
	save_file_dialog.file_selected.connect(_on_file_selected)
	capture = false

func request_capture() -> void:

	if sub_viewport == null:
		push_error("SubViewportScreenshotCapture: SubViewport not assigned")
		return

	var image := sub_viewport.get_texture().get_image()
	if image == null:
		push_error("SubViewportScreenshotCapture: failed to get image from viewport texture")
		return

	_pending_image = image
	save_file_dialog.popup_centered()

func _on_file_selected(path: String) -> void:
	var err := _pending_image.save_png(path)
	if err != OK:
		push_error("SubViewportScreenshotCapture: failed to save to '%s' (%s)" % [path, error_string(err)])
	else:
		print("SubViewportScreenshotCapture: saved to %s" % path)
	
	_pending_image = null
