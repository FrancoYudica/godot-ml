extends Control

@export var _models: Array[ONNXResource] = []
@export var _option: OptionButton
@onready var _processor := $MLPostProcessor


func _ready() -> void:
	
	_option.select(0)
	_processor.setup_model(_models[0])
	
	_option.item_selected.connect(
		func(index: int):
			
			if _models.size() - 1 < index:
				push_warning("Unable to get model of index %s. Out of index" % str(index))
				return
				
			_processor.setup_model(_models[index])
	)
