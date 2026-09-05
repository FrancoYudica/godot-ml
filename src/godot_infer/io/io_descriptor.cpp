#include "io_descriptor.hpp"

namespace ml::Utils {
std::string input_handler_name(const InputType& type) {
    switch (type) {
    case InputType::Texture2D:
        return "Texture2DInputDesc";
    case InputType::FloatArray:
        return "FloatArrayInputDesc";
    }
    return "UnknownInputDesc";
}
std::string output_handler_name(const OutputType& type) {
    switch (type) {
    case OutputType::Texture2D:
        return "Texture2DOutputDesc";
    case OutputType::FloatArray:
        return "FloatArrayOutputDesc";
    }
    return "UnknownOutputDesc";
}
} // namespace ml::Utils