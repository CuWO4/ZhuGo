#ifndef EYE_H_
#define EYE_H_

extern "C" {
  #include "board.h"
}

bool is_eye(Board* board, int row, int col, Piece player);

#endif