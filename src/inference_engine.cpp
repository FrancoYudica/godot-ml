#include "inference_engine.h"
#include "ml_inference/ml_parser.hpp"
#include "ml_inference/ml_utils.hpp"

void godot::MLInferenceEngine::_bind_methods() {
  ClassDB::bind_method(D_METHOD("setup", "model_path"),
                       &MLInferenceEngine::setup);
}

void godot::MLInferenceEngine::setup(String model_path) {
  print_line("Setting up ML inference engine with model: " + model_path);

  ml::Graph graph;
  bool success = ml::Parser::parse(model_path.utf8().ptr(), graph);

  if (!success)
    return;

  ml::Utils::print(graph);
}
