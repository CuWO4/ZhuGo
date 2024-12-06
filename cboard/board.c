#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdint.h>

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

Board* new_board(int rows, int cols);
void delete_board(Board* board);
Board* clone_board(Board* board);
Piece get_piece(Board* board, int row, int col);
unsigned get_qi(Board* board, int row, int col);
void get_random_qi_pos(unsigned* pos_row, unsigned* pos_col, Board* board, int row, int col);
bool is_valid_move(Board* board, int row, int col, Piece player);
void place_piece(Board* board, int row, int col, Piece player);
uint64_t hash(Board* board);


static int dx[4] = { -1, 1, 0, 0 };
static int dy[4] = { 0, 0, -1, 1 };

inline Piece other(Piece piece) {
  assert(piece != EMPTY);
  if (piece == BLACK) {
    return WHITE;
  }
  else {
    return BLACK;
  }
}

uint64_t hash_murmur(uint64_t x) {
  uint64_t h = x;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h * 2685821657736338717ULL;
}

uint64_t hash_piece(int row, int col, Piece player) {
  return hash_murmur(((uint64_t) row << 48) | ((uint64_t) col << 24) | ((uint64_t) player << 4));
}

Board* new_board(int rows, int cols) {
  Board* board = (Board*) malloc(sizeof(Board));

  board->rows = rows;
  board->cols = cols;

  board->mem_size = rows * cols * (sizeof(Piece) + sizeof(unsigned) + sizeof(uint32_t));
  board->mem = calloc(board->mem_size, 1);

  board->is_qi_updated = false;

  int offset = 0;

  board->pieces = (Piece**) malloc(rows * sizeof(Piece*));
  for (int i = 0; i < rows; i++) {
    board->pieces[i] = (Piece*) ((char*) board->mem + offset) + i * cols;
  }

  offset += rows * cols * sizeof(Piece);

  board->qi_cache = (unsigned**) malloc(rows * sizeof(unsigned*));
  for (int i = 0; i < rows; i++) {
    board->qi_cache[i] = (unsigned*) ((char*) board->mem + offset) + i * cols;
  }

  offset += rows * cols * sizeof(unsigned);

  board->qi_pos = (uint32_t**) malloc(rows * sizeof(uint32_t*));
  for (int i = 0; i < rows; i++) {
    board->qi_pos[i] = (uint32_t*) ((char*) board->mem + offset) + i * cols;
  }

  offset += rows * cols * sizeof(uint32_t);
  assert(offset == board->mem_size);

  board->zobrist_hash = 0;

  return board;
}

void delete_board(Board* board) {
  free(board->mem);
  free(board->pieces);
  free(board->qi_cache);
  free(board->qi_pos);
  free(board);
}

Board* clone_board(Board* board) {
  Board* cloned = new_board(board->rows, board->cols);
  memcpy(cloned->mem, board->mem, board->mem_size);
  cloned->zobrist_hash = board->zobrist_hash;
  cloned->is_qi_updated = board->is_qi_updated;
  return cloned;
}

inline bool in_board(Board* board, int row, int col) {
  return 0 <= row && row < board->rows && 0 <= col && col < board->cols;
}

inline Piece get_piece(Board* board, int row, int col) {
  assert(in_board(board, row, col));
  return board->pieces[row][col];
}

inline void add_piece(Board* board, int row, int col, Piece player) {
  assert(get_piece(board, row, col) == EMPTY);
  board->pieces[row][col] = player;
  board->zobrist_hash ^= hash_piece(row, col, player);
  board->is_qi_updated = false;
}

inline void remove_piece(Board* board, int row, int col) {
  Piece piece = get_piece(board, row, col);
  assert(piece != EMPTY);
  board->zobrist_hash ^= hash_piece(row, col, piece);
  board->pieces[row][col] = EMPTY;
  board->is_qi_updated = false;
}

void update_qi(Board* board);

unsigned get_qi(Board *board, int row, int col) {
  if (! board->is_qi_updated) {
    update_qi(board);
  }

  return board->qi_cache[row][col];
}

void get_random_qi_pos(unsigned* pos_row, unsigned* pos_col, Board* board, int row, int col) {
  if (! board->is_qi_updated) {
    update_qi(board);
  }

  uint32_t qi_pos = board->qi_pos[row][col];
  *pos_row = (qi_pos >> 16) & 0xFFFF;
  *pos_col = qi_pos & 0xFFFF;
}

unsigned update_get_qi(
  unsigned* pos_row, unsigned* pos_col,
  Board* board, int row, int col,
  bool** is_piece_visited, bool** is_qi_visited
) {
  assert(in_board(board, row, col));

  if (is_piece_visited[row][col]) {
    return 0;
  }
  is_piece_visited[row][col] = true;

  Piece piece = get_piece(board, row, col);
  assert(piece != EMPTY);

  unsigned qi = 0;

  for (int k = 0; k < 4; k++) {
    int r = row + dx[k];
    int c = col + dy[k];

    if (!in_board(board, r, c)) {
      continue;
    }

    Piece neighbor = get_piece(board, r, c);

    if (neighbor == EMPTY) {
      if (! is_qi_visited[r][c]) {
        qi++;
        *pos_row = r;
        *pos_col = c;
      }
      is_qi_visited[r][c] = true;
    }

    if (neighbor == piece && ! is_piece_visited[r][c]) {
      qi += update_get_qi(pos_row, pos_col, board, r, c, is_piece_visited, is_qi_visited);
    }
  }

  return qi;
}

void update_set_qi(
  Board* board, int row, int col,
  unsigned qi, unsigned pos_row, unsigned pos_col,
  bool** is_piece_set
) {
  assert(in_board(board, row, col));

  if (is_piece_set[row][col]) {
    return;
  }
  is_piece_set[row][col] = true;

  Piece piece = get_piece(board, row, col);
  assert(piece != EMPTY);

  board->qi_cache[row][col] = qi;
  board->qi_pos[row][col] = ((pos_row & 0xFFFF) << 16) | (pos_col & 0xFFFF);

  for (int k = 0; k < 4; k++) {
    int r = row + dx[k];
    int c = col + dy[k];

    if (!in_board(board, r, c)) {
      continue;
    }

    Piece neighbor = get_piece(board, r, c);

    if (neighbor == piece && ! is_piece_set[r][c]) {
      update_set_qi(board, r, c, qi, pos_row, pos_col, is_piece_set);
    }
  }
}

void update_qi(Board* board) {
  #ifndef MAX_N
  #define MAX_N 32
  #endif

  assert(board->rows < MAX_N && board->cols < MAX_N);

  static bool is_piece_set_mem[MAX_N * MAX_N];
  static bool is_piece_visited_mem[MAX_N * MAX_N];
  static bool is_qi_visited_mem[MAX_N * MAX_N];

  static bool* is_piece_set[MAX_N];
  static bool* is_piece_visited[MAX_N];
  static bool* is_qi_visited[MAX_N];

  for (int i = 0; i < board->rows; i++) {
    is_piece_set[i] = is_piece_set_mem + i * board->cols;
    is_piece_visited[i] = is_piece_visited_mem + i * board->cols;
    is_qi_visited[i] = is_qi_visited_mem + i * board->cols;
  }

  memset(is_piece_set_mem, 0, board->rows * board->cols * sizeof(bool));

  for (int r = 0; r < board->rows; r++) {
    for (int c = 0; c < board->cols; c++) {
      if (is_piece_set[r][c]) {
        continue;
      }

      if (get_piece(board, r, c) == EMPTY) {
        continue;
      }

      unsigned qi, pos_row, pos_col;

      memset(is_piece_visited_mem, 0, board->rows * board->cols * sizeof(bool));
      memset(is_qi_visited_mem, 0, board->rows * board->cols * sizeof(bool));

      qi = update_get_qi(&pos_row, &pos_col, board, r, c, is_piece_visited, is_qi_visited);

      update_set_qi(board, r, c, qi, pos_row, pos_col, is_piece_set);
    }
  }

  board->is_qi_updated = true;

  #undef MAX_N
}

bool is_self_capture(Board* board, int row, int col, Piece player) {
  assert(in_board(board, row, col));
  assert(get_piece(board, row, col) == EMPTY);

  for (int k = 0; k < 4; k++) {
    int r = row + dx[k];
    int c = col + dy[k];

    if (!in_board(board, r, c)) {
      continue;
    }

    Piece neighbor = get_piece(board, r, c);

    if (neighbor == EMPTY) {
      /* have at least one qi remain after play */
      return false;
    }

    if (neighbor != player && get_qi(board, r, c) <= 1) {
      /* kill opponents's piece */
      return false;
    }

    if (neighbor == player && get_qi(board, r, c) > 1) {
      /* after merging, have at least one qi */
      return false;
    }
  }

  return true;
}

bool is_valid_move(Board* board, int row, int col, Piece player) {
  assert(in_board(board, row, col));

  Piece piece = get_piece(board, row, col);
  if (piece != EMPTY) {
    return false;
  }

  return ! is_self_capture(board, row, col, player);
}

void remove_string(Board* board, int row, int col) {
  assert(in_board(board, row, col));

  Piece piece = get_piece(board, row, col);

  assert(piece != EMPTY);

  remove_piece(board, row, col);

  for (int k = 0; k < 4; k++) {
    int r = row + dx[k];
    int c = col + dy[k];

    if (!in_board(board, r, c)) {
      continue;
    }

    Piece neighbor = get_piece(board, r, c);

    if (neighbor == piece) {
      remove_string(board, r, c);
    }
  }
}

void place_piece(Board* board, int row, int col, Piece player) {
  assert(is_valid_move(board, row, col, player));
  add_piece(board, row, col, player);

  for (int k = 0; k < 4; k++) {
    int r = row + dx[k];
    int c = col + dy[k];

    if (!in_board(board, r, c)) {
      continue;
    }

    Piece neighbor = get_piece(board, r, c);

    if (neighbor == EMPTY) {
      continue;
    }

    if (neighbor != player && get_qi(board, r, c) == 0) {
      remove_string(board, r, c);
    }
  }
}

uint64_t hash(Board* board) {
  return board->zobrist_hash;
}