#ifndef BGTF_H__
#define BGTF_H__

#include <cstdint>
#include <vector>
#include <string>

/*
 *                                BGTF (Binary Go Train Format)
 *
 *                                   description                  type
 *                      +-------------------------------------+------------+
 *                      |      magic number (0x3456789A)      |   uint32   |
 *                      +-------------------------------------+------------+
 *                      |            version (0)              |   uint32   |
 *                      +-------------------------------------+------------+
 *                      |            - reserved -             |     64B    |
 *                      +-------------------------------------+------------+
 *                      |             game count              |   uint32   |
 *               +----> +-------------------------------------+------------+
 *               |      |          game offset [0]            |   uint64   | -----------------+
 *               |      |          game offset [1]            |   uint64   | ---------------+ |
 *               |      |          game offset [2]            |   uint64   | -------------+ | |
 *   game count  |      |                 .                   |      .     |   .          | | |
 *    in total   |      |                 .                   |      .     |   .          | | |
 *               |      |                 .                   |      .     |   .          | | |
 *               |      |    game offset [game count - 1]     |   uint64   | -----+       | | |
 *               +----> +-------------------------------------+------------+      |       | | |
 *                      |             - padding -             |            |      |       | | |
 *                      +-------------------------------------+------------+ <----|-------|-|-+
 *                      |               game [0]              |    Game    |      |       | |
 *                      +-------------------------------------+------------+      |       | |
 *                      |             - padding -             |            |      |       | |
 *                      +-------------------------------------+------------+ <----|-------|-+
 *                      |               game [1]              |    Game    |      |       |
 *                      +-------------------------------------+------------+      |       |
 *                      |             - padding -             |            |      |       |
 *                      +-------------------------------------+------------+ <----|-------+
 *                      |               game [2]              |    Game    |      |
 *                      +-------------------------------------+------------+      |
 *                      |                 .                   |            |      |
 *                      |                 .                   |            |      |
 *                      |                 .                   |            |      |
 *                      +-------------------------------------+------------+      |
 *                      |             - padding -             |            |      |
 *                      +-------------------------------------+------------+ <----+
 *                      |        game [game count - 1]        |    Game    |
 *                      +-------------------------------------+------------+
 *
 *                                           Game
 *                      +-------------------------------------+------------+
 *                      |             turn count              |   uint32   |
 *                      +-------------------------------------+------------+
 *                      |  winner (0=black, 1=white, 2=draw)  |   uint32   |
 *               +----> +-------------------------------------+------------+
 *               |      |              turn [0]               |    Turn    |
 *               |      |              turn [1]               |    Turn    |
 *               |      |              turn [2]               |    Turn    |
 *   turn count  |      |                 .                   |      .     |
 *    in total   |      |                 .                   |      .     |
 *               |      |                 .                   |      .     |
 *               |      |        turn [turn count - 1]        |    Turn    |
 *               +----> +-------------------------------------+------------+
 *
 *                                           Turn
 *                      +-------------------------------------+------------+
 *                      |       move row {0, 1, ..., 18}      |   uint16   |
 *                      +-------------------------------------+------------+
 *                      |     move column {0, 1, ..., 18}     |   uint16   |
 *                      +-------------------------------------+------------+
 *                      |     target policy distribution      | float[362] |
 *                      +-------------------------------------+------------+

 *
 *  Note:   - the board is 0-based.
 *          - let N be the count of invalid coordinates in a move. a coordinate (row or column)
 *            is invalid if it's not in the range [0, 19) (i.e., < 0 or ≥ 19). a move will be 
 *            treated as:
 *              <​N = 0> a valid move,
 *              <​N = 1> passing turn,
 *              <​N = 2> resigning.
 *          - target policy target is row-first arranged 19x19 [0:360] + pass [361],
 *            which means the index of 0-based (row, col) is row * 19 + col.
 *          - padding depends on implementation. best practice to access game is through
 *            offset table at header. except for the padding position, there is no padding 
 *            between fields.
 *          - when hash type is unsupported (custom), hash check is skipped.
 */

namespace BGTF { /* binary go train format */
  static const uint32_t head_magic_number = 0x3456789A;

  struct Turn {
    uint16_t row, col;
    float policy_distribution[362];
  };

  enum Winner { Black = 0, White = 1, Draw = 2 };

  struct Game {
    uint32_t winner;
    std::vector<Turn> turns;
  };

  class BGTF {
  public:
    BGTF(const std::vector<Game>& games);

    explicit BGTF(std::string path);

    void save_to(std::string path);

  private:
    std::vector<Game> games;
  };
}

#endif
