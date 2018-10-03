// Create at 2017-03-27

#include "symbol_table.h"

#include <assert.h>
#include <stdlib.h>
#include "util.h"

namespace pocketkaldi {

char kBosSymbol[] = "<s>";
char kEosSymbol[] = "</s>";

SymbolTable::SymbolTable(): bos_id_(0), eos_id_(0) {}

Status SymbolTable::Read(const std::string &filename) {
  util::ReadableFile fd;
  PK_CHECK_STATUS(fd.Open(filename));

  Status status = Status::OK();
  std::string line;
  words_.reserve(65536);
  while (fd.ReadLine(&line, &status) && status.ok()) {
    std::vector<std::string> fields = util::Split(line, " ");
    if (fields.size() != 2) {
      return Status::Corruption(util::Format(
          "2 column expected but {} found: {}",
          fields.size(),
          line));
    }

    std::string word = fields[0];
    long word_id = 0;
    PK_CHECK_STATUS(util::StringToLong(fields[1], &word_id));
    
    word_ids_[word] = word_id;
    if (word_id >= words_.size()) words_.resize(word_id + 1);
    words_[word_id] = word;
  }
  PK_CHECK_STATUS(status);

  // Find BOS and EOS ids
  if (word_ids_.find(kBosSymbol) == word_ids_.end() ||
      word_ids_.find(kEosSymbol) == word_ids_.end()) {
    return Status::Corruption("symbol_table: unable to find BOS/EOS symbol");
  }
  bos_id_ = word_ids_[kBosSymbol];
  eos_id_ = word_ids_[kEosSymbol];

  return Status::OK();
}

const char *SymbolTable::Get(int symbol_id) const {
  assert(symbol_id < words_.size() && "symbol_id out of boundary");
  return words_[symbol_id].c_str();
}

int SymbolTable::GetId(const std::string &word) const {
  std::unordered_map<std::string, int>::const_iterator it = word_ids_.find(word);
  if (it == word_ids_.end()) {
    return kNotExist;
  } else {
    return it->second;
  }
}

}  // namespace pocketkaldi
