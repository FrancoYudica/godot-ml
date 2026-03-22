#include "../ml_parser.hpp"
#include "../ml_utils.hpp"
#include <iostream>

using namespace ml;

int main(int argc, char *argv[]) {

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <onnx_file.onnx>\n";
    return 1;
  }

  Graph graph = Parser::parse(argv[1]);
  Utils::print(graph);
  return 0;
}