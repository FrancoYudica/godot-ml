@tool
extends EditorImportPlugin

enum Presets { DEFAULT }

func _get_importer_name():
	return "godot_infer.onnx"

func _get_visible_name():
	return "ONNX Model"

func _get_recognized_extensions():
	return ["onnx"]

func _get_save_extension():
	return "res"

func _get_resource_type():
	return "ONNXResource"

func _get_preset_count():
	return Presets.size()

func _get_preset_name(preset_index):
	match preset_index:
		Presets.DEFAULT:
			return "Default"
		_:
			return "Unknown"

func _get_import_options(path, preset_index):
	match preset_index:
		Presets.DEFAULT:
			return [{
					   "name": "merge_operators",
					   "default_value": true,
					   "hint_string": "Set to true for merging compatible operators into a single kernel"
					}]
		_:
			return []

func _import(source_file, save_path, options, platform_variants, gen_files):
	var resource = ONNXResource.load_from_file(source_file)
	if resource == null:
		return ERR_CANT_OPEN
	return ResourceSaver.save(resource, "%s.%s" % [save_path, _get_save_extension()])
