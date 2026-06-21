extends PanelContainer

@export var name_label: Label
@export var avg_label: Label
@export var min_label: Label
@export var max_label: Label

var _min: float = 1e8
var _avg: float = 0.0
var _max: float = 0.0
var _count: int = 0
var operator_name: String:
	set(value):
		name_label.text = value


func new_report(duration_ms: float):
	
	_count += 1
	var countf = float(_count)
	_avg = (_avg * (countf - 1) + duration_ms) / countf
	_min = minf(duration_ms, _min)
	_max = maxf(duration_ms, _max)
	avg_label.text = "%.3f" % _avg
	min_label.text = "%.3f" % _min
	max_label.text = "%.3f" % _max
