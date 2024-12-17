#ifndef BOARD_H_
#define BOARD_H_

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

enum {
  EMPTY = 0,
  BLACK = 1,
  WHITE = 2,
};
typedef char Piece;

typedef struct {
  int rows;
  int cols;

  size_t mem_size;
  void* mem;

  Piece** pieces;
  bool is_qi_updated;
  unsigned** qi_cache;
  uint32_t** qi_pos;
  uint64_t zobrist_hash;
} Board;

bool in_board(Board* board, int row, int col);
Board* new_board(int rows, int cols);
void delete_board(Board* board);
Board* clone_board(Board* board);
Piece get_piece(Board* board, int row, int col);
unsigned get_qi(Board* board, int row, int col);
void get_random_qi_pos(unsigned* pos_row, unsigned* pos_col, Board* board, int row, int col);
bool is_valid_move(Board* board, int row, int col, Piece player);
void place_piece(Board* board, int row, int col, Piece player);
uint64_t hash(Board* board);


inline bool in_board(Board* board, int row, int col) {
  return 0 <= row && row < board->rows && 0 <= col && col < board->cols;
}

inline Piece get_piece(Board* board, int row, int col) {
  assert(in_board(board, row, col));
  return board->pieces[row][col];
}

#endif