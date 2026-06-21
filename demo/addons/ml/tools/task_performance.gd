extends Control

@export var save_file_dialog: FileDialog
@export var clear_button: Button
@export var export_button: Button
@export var collecting_reports_check: CheckBox
@export var entries_parent: Control
@export var entry_scene: PackedScene
@export var dispatch_count: Label
@export var avg_dispatch_time: Label
@export var min_dispatch_time: Label
@export var max_dispatch_time: Label

var _entries: Dictionary[String, Control]
var _average_time_per_dispatch: float = 0.0
var _dispatch_count: int = 0
var _min: float = 1e10
var _max: float = 0.0

var _reports: Array[Dictionary] = []
var _collecting: bool = true

func _ready() -> void:
	
	clear_button.pressed.connect(_clear_entries)
	export_button.pressed.connect(_export_request)
	save_file_dialog.file_selected.connect(_file_selected)
	collecting_reports_check.toggled.connect(
		func(toggled_on):
			_collecting = toggled_on
	)
	_clear_entries()

func push_task_report(data: Dictionary):
	if not _collecting:
		_refresh_ui()
		return
	
	if data.is_empty():
		return
		
	_reports.append(data)
	_dispatch_count += 1
	var countf = float(_dispatch_count)
	_average_time_per_dispatch = (_average_time_per_dispatch * (countf - 1) + data.total_gpu_time_ms) / countf
	_min = minf(data.total_gpu_time_ms, _min)
	_max = maxf(data.total_gpu_time_ms, _max)
	
	_refresh_ui()

	for operator in data.operators:
		var entry = _get_or_create_entry(operator.name)
		_update_entry(entry, operator.duration_ms)

func _refresh_ui():
	dispatch_count.text = "%s" % _dispatch_count
	avg_dispatch_time.text = "%.3f" % _average_time_per_dispatch
	min_dispatch_time.text = "%.3f" % _min
	max_dispatch_time.text = "%.3f" % _max

func _get_or_create_entry(operator_name: String):
	if not _entries.has(operator_name):
		var entry = entry_scene.instantiate()
		entry.operator_name = operator_name
		entries_parent.add_child(entry)
		_entries[operator_name] = entry
		
	return _entries[operator_name]

func _update_entry(entry, duration: float):
	entry.new_report(duration)
	

func _clear_entries():
	while entries_parent.get_child_count() > 0:
		entries_parent.remove_child(entries_parent.get_child(0))
	_dispatch_count = 0
	_average_time_per_dispatch = 0.0
	_min = 20
	_max = 0.0
	_entries.clear()
	_reports.clear()

func _export_request():
	save_file_dialog.popup_centered()


func _file_selected(filename: String):
	_export_to(filename)

func _export_to(filename: String):
	print("Exporting to file %s" % filename)
	
	var summary = {}
	summary["samples"] = _dispatch_count
	summary["avg_dispatch"] = _average_time_per_dispatch
	summary["min_dispatch"] = _min
	summary["max_dispatch"] = _max
	
	var data = {}
	data["summary"] = summary
	data["reports"] = _reports
	
	var file := FileAccess.open(filename, FileAccess.WRITE)
	if file == null:
		push_error("TaskPerformance: could not open '%s' for writing (error %s)" % [
			filename, error_string(FileAccess.get_open_error())])
		return
	file.store_string(JSON.stringify(data, "\t"))
	file.close()
	print("TaskPerformance: exported %d reports to %s" % [_dispatch_count, filename])
