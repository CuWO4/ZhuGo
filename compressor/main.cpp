#include "compressor.h"

#include <cassert>

int main(int argc, char** argv) {
  assert(argc == 3 && "need two command line arguments [input file path] & [output file path]");
  compress(argv[1], argv[2]);
}
