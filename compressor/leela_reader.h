#ifndef LEELA_READER_H__
#define LEELA_READER_H__

#include <cstdio>

enum Player { black = 0, white = 1 };

bool get_from_leela_go_format(
  FILE* in,
  bool& new_game,
  Player& player,
  bool& is_won,
  bool self_stones[],
  bool oppo_stones[],
  float distribution[]
);

#endif
