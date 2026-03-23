#ifndef ML_INFERENCE_ENGINE_H
#define ML_INFERENCE_ENGINE_H
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>

namespace godot {
class MLInferenceEngine : public RefCounted {
  GDCLASS(MLInferenceEngine, RefCounted)

public:
  void setup(String model_path);

protected:
  static void _bind_methods();
};
} // namespace godot

#endif
