extends PanelContainer

@export var name_label: Label
@export var avg_label: Label
@export var min_label: Label
@export var max_label: Label
@export var ratio_label: Label
@export var usage_color: ColorRect

var _min: float = 1e8
var _avg: float = 0.0
var _max: float = 0.0
var _count: int = 0

var operator_name: String:
	set(value):
		name_label.text = value

var avg: float:
	get:
		return _avg

func set_ratio(r: float):
	var from_color = Color.GREEN
	var to_color = Color.RED
	# Interpolamos cada componente HSV por separado
	var t = sqrt(sqrt(r))
	var h = lerp(from_color.h, to_color.h, t)
	var s = lerp(from_color.s, to_color.s, t)
	var v = lerp(from_color.v, to_color.v, t)
	
	# Reconstruimos el color a partir del nuevo espacio HSV
	usage_color.color = Color.from_hsv(h, s, v)
	ratio_label.text = "%.2f" % (r * 100.0)
	
func new_report(duration_ms: float):
	
	_count += 1
	var countf = float(_count)
	_avg = (_avg * (countf - 1) + duration_ms) / countf
	_min = minf(duration_ms, _min)
	_max = maxf(duration_ms, _max)
	avg_label.text = "%.3f" % _avg
	min_label.text = "%.3f" % _min
	max_label.text = "%.3f" % _max
