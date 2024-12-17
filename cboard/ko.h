#ifndef KO_H_
#define KO_H_

#include "board.h"

#include <stdbool.h>

bool does_violate_ko(Board* board, Piece player, int row, int col, Board* last_board);

#endif