#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "compressor.h"

#include "leela_reader.h"
#include "bgtf.h"

int compress(std::string in_path, std::string out_path) {
  std::vector<BGTF::Game> games;

  FILE* in = std::fopen(in_path.c_str(), "r");

  if (!in) {
    fprintf(stderr, "failed to open the file %s\n", in_path.c_str());
    return -1;
  }

  bool new_game;
  Player player;
  bool is_won;
  static bool buffer[1200] = {};
  bool* self_stones = &buffer[0];
  bool* oppo_stones = &buffer[400];
  bool* last_turn_self_stones = &buffer[800];
  BGTF::Turn current_turn;

  while (get_from_leela_go_format(
    in,
    new_game,
    player,
    is_won,
    self_stones,
    oppo_stones,
    current_turn.policy_distribution
  )) {
    if (new_game) {
      games.push_back(BGTF::Game());
      games.back().winner = is_won ? BGTF::Black : BGTF::White;

      current_turn.row = current_turn.col = 19; // new game
    }
    else {
      int diff_idx = -1;
      for (int i = 0; i < 361; i++) {
        if (oppo_stones[i] ^ last_turn_self_stones[i]) {
          diff_idx = i;
          break;
        }
      }

      if (diff_idx < 0) { /* pass turn */
        current_turn.row = current_turn.col = 20;
      }
      else {
        current_turn.row = diff_idx / 19;
        current_turn.col = diff_idx % 19;
      }
    }

    games.back().turns.push_back(current_turn);

    std::swap(self_stones, last_turn_self_stones);
  }

  BGTF::BGTF file(std::move(games));
  file.save_to(out_path);

  return 0;
}
