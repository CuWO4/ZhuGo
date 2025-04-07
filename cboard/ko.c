#include "ko.h"

#include <assert.h>

inline bool is_single_piece(Board* board, int row, int col) {
  Piece piece = get_piece(board, row, col);

  assert(piece != EMPTY);

  for_neighbor(board, row, col, r, c) {
    if (!in_board(board, r, c)) {
      continue;
    }

    if (get_piece(board, r, c) == piece) {
      return false;
    }
  }

  return true;
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

  for_neighbor(board, row, col, r, c) {
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