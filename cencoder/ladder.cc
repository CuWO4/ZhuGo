#include "ladder.h"

#include <stack>
#include <tuple>

constexpr int max_size = 19;

std::set<Point> get_go_string(Board* board, int row, int col) {
  Piece piece = get_piece(board, row, col);
  if (piece == EMPTY) {
    return {};
  }

  std::stack<Point> to_be_visited {};
  std::set<Point> go_string {};

  to_be_visited.push({row, col});

  while (!to_be_visited.empty()) {
    auto pos = to_be_visited.top();
    to_be_visited.pop();

    if (go_string.count(pos) > 0) {
      continue;
    }
    go_string.insert(pos);

    for_neighbor(board, row, col, r, c) {
      if (!in_board(board, r, c)) {
        continue;
      }

      if (get_piece(board, r, c) == piece) {
        to_be_visited.push({r, c});
      }
    }
  }

  return go_string;
}

/**
 * @return  is_captured<bool>, escape_steps<int>, escape_path<set>
 */
std::tuple<bool, int, std::set<Point>>
analyze_go_string(Board* board, int row, int col) {
  assert(in_board(board, row, col));

  Piece piece = get_piece(board, row, col);
  assert(piece != EMPTY);

  if (get_qi(board, row, col) > 1) {
    return { false, 0, {} };
  }

  Piece escaping_player = piece;
  Piece chasing_player = other(escaping_player);

  unsigned escape_r, escape_c;
  get_random_qi_pos(&escape_r, &escape_c, board, row, col);

  if (!is_valid_move(board, escape_r, escape_c, escaping_player)) {
    return { true, 0, {} };
  }

  // When compiling on Windows, msvc will place the placement after get_qi out of order, 
  // causing the logic to crash; add the volatile modifier to prevent the ordering
  // Fuck you, msvc. Go eating shit.
  volatile Board* __board = board; 
  place_piece((Board*) __board, escape_r, escape_c, escaping_player);

  if (get_qi(board, escape_r, escape_c) > 2) {
    return { false, 0, {} };
  }

  if (get_qi(board, escape_r, escape_c) < 2) {
    return { true, 1, { { escape_r, escape_c } } };
  }

  bool is_captured = false;
  int escape_steps = 0;
  std::set<Point> escape_path = {};

  for_neighbor(board, escape_r, escape_c, chase_r, chase_c) {
    if (
      ! in_board(board, chase_r, chase_c)
      || ! is_valid_move(board, chase_r, chase_c, chasing_player)
    ) {
      continue;
    }

    Board* after_chasing_board = clone_board(board);
    volatile Board* __after_chasing_board = after_chasing_board;
    place_piece((Board*) __after_chasing_board, chase_r, chase_c, chasing_player);

    bool is_opponents_string_less_than_2_qi = false;
    for_neighbor(after_chasing_board, escape_r, escape_c, r, c) {
      if (
        !in_board(after_chasing_board, r, c)
        || get_piece(after_chasing_board, r, c) != chasing_player
      ) {
        continue;
      }

      if (get_qi(after_chasing_board, r, c) < 2) {
        is_opponents_string_less_than_2_qi = true;
        break;
      }

    }
    if (is_opponents_string_less_than_2_qi) { // escaped
      continue;
    }

    auto [sub_is_captured, sub_escape_steps, sub_escape_path] =
      analyze_go_string(after_chasing_board, escape_r, escape_c);

    if (sub_is_captured) {
      is_captured = true;
      escape_steps = std::max(escape_steps, sub_escape_steps + 1);
      escape_path.insert({ escape_r, escape_c });
      escape_path.insert(sub_escape_path.begin(), sub_escape_path.end());
    }

    delete_board(after_chasing_board);
  }

  return { is_captured, escape_steps, escape_path };
}

LadderAnalysis::LadderAnalysis(Board* board, int threshold) {
  assert(board->rows < max_size && board->cols < max_size);

  std::set<std::pair<int, int>> visited {};

  for_each_cell(board, r, c) {
    if (visited.count({r, c}) > 0 || get_piece(board, r, c) == EMPTY) {
      continue;
    }

    auto go_string = get_go_string(board, r, c);
    visited.insert(go_string.begin(), go_string.end());

    Board* cloned = clone_board(board);
    auto [is_captured, escape_steps, escape_path] = analyze_go_string(cloned, r, c);
    delete_board(cloned);

    if (is_captured && escape_steps > threshold) {
      this->trapped_stones.insert(go_string.begin(), go_string.end());
      this->escape_path.insert(escape_path.begin(), escape_path.end());
    }
  }
}

bool LadderAnalysis::is_stone_trapped(int row, int col) {
  return this->trapped_stones.count({row, col}) > 0;
}

bool LadderAnalysis::is_stone_on_escape_path(int row, int col) {
  return this->escape_path.count({row, col}) > 0;
}