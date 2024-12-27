#ifndef LADDER_H_
#define LADDER_H_

extern "C" {
  #include "board.h"
}

#include <set>

using Point = std::pair<int, int>;

class LadderAnalysis {
public:
  LadderAnalysis(Board* board, int threshold);

  bool is_stone_trapped(int row, int col);
  bool is_stone_on_escape_path(int row, int col);

private:

  std::set<Point> trapped_stones;
  std::set<Point> escape_path;
};

#endif