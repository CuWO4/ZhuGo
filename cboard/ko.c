#include "ko.h"

#include <assert.h>

static int dx[4] = { 1, -1, 0, 0 };
static int dy[4] = { 0, 0, 1, -1 };

inline bool is_single_piece(Board* board, int row, int col) {
  Piece piece = get_piece(board, row, col);

  assert(piece != EMPTY);

  for (int k = 0; k < 4; k++) {
    int r = row + dx[k];
    int c = col + dy[k];

    if (!in_board(board, r, c)) {
      continue;
    }

    if (get_piece(board, r, c) == piece) {
      return false;
    }
  }
  
  return true;
}

inline Piece other(Piece player) {
  assert(player != EMPTY);

  return player == BLACK ? WHITE : BLACK;
}

bool does_violate_ko(Board *board, Piece player, int row, int col, Board *last_board) {
  assert(last_board != NULL);
  assert(in_board(board, row, col));

  if (!is_valid_move(board, row, col, player)) {
    return false;
  }

  if (
    get_piece(last_board, row, col) != player 
    || !is_single_piece(last_board, row, col)
  ) {
    return false;
  }

  for (int k = 0; k < 4; k++) {
    int r = row + dx[k];
    int c = col + dy[k];

    if (!in_board(board, r, c)) {
      continue;
    }

    if (
      get_piece(board, r, c) == other(player)
      && is_single_piece(board, r, c)
      && get_qi(board, r, c) == 1
    ) {
      return true;
    }
  }

  return false;
}