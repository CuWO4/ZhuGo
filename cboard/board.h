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

inline Piece other(Piece player) {
  assert(player != EMPTY);
  return player == BLACK ? WHITE : BLACK;
}

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

static const int __dx[] = { -1, 1, 0 ,0 };
static const int __dy[] = { 0, 0, -1, 1 };

#ifndef for_each_cell
  #define for_each_cell(board, r, c) \
    for (int r = 0; r < board->rows; r++) \
      for (int c = 0; c < board->cols; c++)
#endif

#ifndef for_neighbor
  #define for_neighbor(board, row, col, r, c) \
    for ( \
      int __k = 0, r, c;\
      r = row + __dx[__k], c = col + __dy[__k], __k < 4; \
      __k++)
#endif

#endif