#include "eye.h"

bool is_eye(Board *board, int row, int col, Piece player) {
  if (get_piece(board, row, col) != EMPTY) {
    return false;
  }

  for_neighbor(board, row, col, r, c) {
    if (!in_board(board, r, c)) {
      continue;
    }

    Piece neighbor = get_piece(board, r, c);

    if (neighbor != player) {
      return false;
    }
  }

  /* walk the corners */
  static const int cdx[] = { -1, -1, 1, 1 };
  static const int cdy[] = { -1, 1, -1, 1 };

  int friendly_corners = 0;
  int off_board_corners = 0;

  for (int k = 0; k < 4; k++) {
    int r = row + cdx[k];
    int c = col + cdy[k];

    if (!in_board(board, r, c)) {
      off_board_corners++;
    }
    else {
      Piece corner = get_piece(board, r, c);
      friendly_corners += corner == player;
    }
  }

  if (off_board_corners > 0) {
    return friendly_corners >= 4 - off_board_corners;
  }
  else {
    return friendly_corners >= 3;
  }
}