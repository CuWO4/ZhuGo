#include "bgtf.h"

#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cassert>

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

namespace BGTF {

BGTF::BGTF(const std::vector<Game>& games) : games(games) {}

BGTF::BGTF(std::vector<Game>&& games) : games(std::move(games)) {}

BGTF::BGTF(std::string path) {
  assert(false && "not implemented");
}

void BGTF::save_to(std::string path) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file for writing");
  }

  struct GameData {
    std::vector<char> data;
    size_t padded_size;
  };

  std::vector<GameData> games_data;
  std::vector<uint64_t> game_offsets;

  const size_t header_size = 4 + 4 + 64 + 4;
  const size_t offset_table_size = games.size() * 8;
  const size_t initial_padding = (8 - (header_size + offset_table_size) % 8) % 8;
  uint64_t current_offset = header_size + offset_table_size + initial_padding;

  for (const auto& game : games) {
    std::vector<char> data;

    uint32_t turn_n_le = game.turns.size();
    data.insert(data.end(), reinterpret_cast<char*>(&turn_n_le), reinterpret_cast<char*>(&turn_n_le) + 4);
    uint32_t winner_le = game.winner;
    data.insert(data.end(), reinterpret_cast<char*>(&winner_le), reinterpret_cast<char*>(&winner_le) + 4);

    for (const auto& turn : game.turns) {
      uint16_t row_le = turn.row;
      data.insert(data.end(), reinterpret_cast<char*>(&row_le), reinterpret_cast<char*>(&row_le) + 2);
      uint16_t col_le = turn.col;
      data.insert(data.end(), reinterpret_cast<char*>(&col_le), reinterpret_cast<char*>(&col_le) + 2);
      const char* policy_ptr = reinterpret_cast<const char*>(turn.policy_distribution);
      data.insert(data.end(), policy_ptr, policy_ptr + 362 * sizeof(float));
    }

    size_t data_size = data.size();
    size_t pad_size = (8 - (data_size % 8)) % 8;
    if (pad_size > 0) {
      data.insert(data.end(), pad_size, 0);
    }

    GameData gd;

    gd.data = std::move(data);
    gd.padded_size = data_size + pad_size;
    games_data.push_back(std::move(gd));

    game_offsets.push_back(current_offset);
    current_offset += gd.padded_size;
  }

  uint32_t magic_le = head_magic_number;
  file.write(reinterpret_cast<const char*>(&magic_le), 4);

  uint32_t version_le = 0;
  file.write(reinterpret_cast<const char*>(&version_le), 4);

  char reserved[64] = {0};
  file.write(reserved, 64);

  uint32_t game_count_le = games.size();
  file.write(reinterpret_cast<const char*>(&game_count_le), 4);

  for (uint64_t offset : game_offsets) {
    uint64_t offset_le = offset;
    file.write(reinterpret_cast<const char*>(&offset_le), 8);
  }

  file.write(reserved, initial_padding);

  for (const auto& gd : games_data) {
    file.write(gd.data.data(), gd.data.size());
  }
}

}