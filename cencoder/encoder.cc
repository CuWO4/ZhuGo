#include "encoder.h"

extern "C" {
  #include "ko.h"
}

#include "eye.h"
#include "ladder.h"

#include <array>

#define ENCODER_FORMAL_ARGS float* data, int& offset, Board* board, Board* last_board, Piece player

static void encode_stone         (ENCODER_FORMAL_ARGS);
static void encode_player        (ENCODER_FORMAL_ARGS);
static void encode_valid_move    (ENCODER_FORMAL_ARGS);
static void encode_qi            (ENCODER_FORMAL_ARGS);
static void encode_qi_after_play (ENCODER_FORMAL_ARGS);
static void encode_ladder        (ENCODER_FORMAL_ARGS);
static void encode_ko            (ENCODER_FORMAL_ARGS);

static constexpr void (* encoders[])(ENCODER_FORMAL_ARGS) = {
  encode_stone,
  encode_player,
  encode_valid_move,
  encode_qi,
  encode_qi_after_play,
  encode_ladder,
  encode_ko,
};

float* encode(Board* board, Board* last_board, Piece player) {
  assert(board->rows == last_board->rows && board->cols == last_board->cols);
  assert(board->rows < max_size && board->cols < max_size);

  float* data = static_cast<float*>(calloc(channels * board->rows * board->cols, sizeof(float)));

  int offset = 0;
  for (const auto& encoder : encoders) {
    encoder(data, offset, board, last_board, player);
  }

  assert(offset == channels * board->rows * board->rows);

  return data;
}

void encode_stone(ENCODER_FORMAL_ARGS) {
  int rows = board->rows;
  int cols = board->cols;
  int plain_size = rows * cols;

  Piece other_player = other(player);

  for_each_cell(board, r, c) {
    Piece piece = get_piece(board, r, c);
    data[offset + 0 * plain_size + r * cols + c] = piece == player;
    data[offset + 1 * plain_size + r * cols + c] = piece == other_player;
  }

  offset += 2 * plain_size;
}

void encode_player(ENCODER_FORMAL_ARGS) {
  int rows = board->rows;
  int cols = board->cols;
  int plain_size = rows * cols;

  float fill = player == BLACK ? 1.0f : 0.0f;
  for (int k = 0; k < plain_size; k++) {
    data[offset + k] = fill;
  }

  offset += plain_size;
}

void encode_valid_move(ENCODER_FORMAL_ARGS) {
  int rows = board->rows;
  int cols = board->cols;
  int plain_size = rows * cols;

  for_each_cell(board, r, c) {
    data[offset + r * cols + c] = 
      is_valid_move(board, r, c, player)
      && ! does_violate_ko(board, player, r, c, last_board)
      && ! is_eye(board, r, c, player);
  }

  offset += plain_size;
}


void encode_qi(ENCODER_FORMAL_ARGS) {
  int rows = board->rows;
  int cols = board->cols;
  int plain_size = rows * cols;

  constexpr int max_qi = 8;

  for (int k = 0; k < max_qi; k++) {
    for_each_cell(board, r, c) {
      data[offset + k * plain_size + r * cols + c] = get_qi(board, r, c) > k;
    }
  }

  offset += max_qi * plain_size;
}


void encode_qi_after_play(ENCODER_FORMAL_ARGS) {
  int rows = board->rows;
  int cols = board->cols;
  int plain_size = rows * cols;

  constexpr int max_qi = 8;
  /* to improve cache performance */
  std::array<std::array<int, max_size>, max_size> qi_after_play = {};

  for_each_cell(board, r, c) {
    if (
      ! is_valid_move(board, r, c, player)
      || does_violate_ko(board, player, r, c, last_board)
      || is_eye(board, r, c, player)
    ) {
      qi_after_play[r][c] = 0;
    }
    else {
      Board* cloned = clone_board(board);
      place_piece(cloned, r, c, player);
      qi_after_play[r][c] = get_qi(cloned, r, c);
      delete_board(cloned);
    }
  }

  for (int k = 0; k < max_qi; k++) {
    for_each_cell(board, r, c) {
      data[offset + k * plain_size + r * cols + c] = qi_after_play[r][c] > k;
    }
  }

  offset += max_qi * plain_size;
}


void encode_ladder(ENCODER_FORMAL_ARGS) {
  int rows = board->rows;
  int cols = board->cols;
  int plain_size = rows * cols;

  constexpr int ladder_channels = 3;

  memset(data + offset, 0, ladder_channels * plain_size * sizeof(*data));

  LadderAnalysis analysis { board, ladder_threshold };

  for_each_cell(board, r, c) {
    if (analysis.is_stone_trapped(r, c)) {
      int player_offset = get_piece(board, r, c) == player ? 0 : plain_size;
      data[offset + player_offset + r * cols + c] = 1.0;
    }
  }

  for_each_cell(board, r, c) {
    data[offset + 2 * plain_size + r * cols + c] =
      analysis.is_stone_on_escape_path(r, c);
  }

  offset += ladder_channels * plain_size;
}


void encode_ko(ENCODER_FORMAL_ARGS) {
  int rows = board->rows;
  int cols = board->cols;
  int plain_size = rows * cols;

  for_each_cell(board, r, c) {
    data[offset + r * cols + c] =
      is_valid_move(board, r, c, player)
      && does_violate_ko(board, player, r, c, last_board);
  }

  offset += plain_size;
}

