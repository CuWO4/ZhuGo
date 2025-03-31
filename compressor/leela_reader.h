#ifndef LEELA_READER_H__
#define LEELA_READER_H__

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstdio>

enum Player { black = 0, white = 1 };

bool get_from_leela_go_format(
  bool& new_game,
  FILE* in,
  Player& player,
  bool& is_won,
  bool self_stones[],
  bool oppo_stones[],
  float distribution[]
);

#endif
