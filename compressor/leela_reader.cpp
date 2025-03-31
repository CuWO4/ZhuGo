#include "leela_reader.h"

#include <cassert>

static int parse_hex(char ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  else if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  else if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
  else {
    fprintf(stderr, "invalid hex digit `%c`\n", ch);
    assert(false);
  }
  return 0;
}

static bool has_stone(char stone_string[], int i) {
  return !!(parse_hex(stone_string[i / 4]) & (1 << (3 - i % 4)));
}

static void get_line(char* buffer, FILE* stream) {
  char ch;
  while ((ch = fgetc(stream)) == '\n');
  *buffer++ = ch;
  while ((ch = fgetc(stream)) != '\n') *buffer++ = ch;
}

static void skip_line(FILE* stream) {
  while (fgetc(stream) == '\n');
  while (fgetc(stream) != '\n');
}

static bool is_end(FILE* fp) {
  char c1 = fgetc(fp);
  char c2 = fgetc(fp);
  ungetc(c2, fp);
  ungetc(c1, fp);
  return c1 == EOF || c2 == EOF;
}


bool get_from_leela_go_format(
  bool& new_game,
  FILE* in,
  Player& player,
  bool& is_won,
  bool self_stones[],
  bool oppo_stones[],
  float distribution[]
) {
  if (is_end(in)) {
    return false;
  }

  char self_stone_string_buf[100] = {};
  char oppo_stone_string_buf[100] = {};

  for (int i = 0; i < 16; i++) {
    if (i == 0) get_line(self_stone_string_buf, in);
    else if (i == 8) get_line(oppo_stone_string_buf, in);
    else skip_line(in);
  }

  new_game = true;

  for (int i = 0; i < 361; i++) {
    self_stones[i] = has_stone(self_stone_string_buf, i);
    oppo_stones[i] = has_stone(oppo_stone_string_buf, i);
    assert(!(self_stones[i] && oppo_stones[i]) && "two piece at same position");
    new_game &= !(self_stones[i] || oppo_stones[i]);
  }

  fscanf(in, "%d", &player);

  for (int i = 0; i < 362; i++) {
    fscanf(in, "%f", &distribution[i]);
  }

  int win;
  fscanf(in, "%d", &win);
  is_won = win > 0;

  return true;
}
