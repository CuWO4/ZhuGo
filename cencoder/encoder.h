#ifndef ENCODER_H_
#define ENCODER_H_

constexpr int channels = 25;
constexpr int max_size = 19;

constexpr int ladder_threshold = 4;

extern "C" {
  #include "board.h"
}

float* encode(Board* board, Board* last_board, Piece player);

#endif