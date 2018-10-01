// Create at 2017-03-27

#ifndef POCKETKALDI_SYMBOL_TABLE_H_
#define POCKETKALDI_SYMBOL_TABLE_H_

#include <string>
#include <vector>
#include <unordered_map>
#include "status.h"
#include "util.h"

#define PK_SYMBOLTABLE_SECTION "SYM0"

namespace pocketkaldi {

// Store a list of symbols. And the symbol string could be got by
//   SymbolTable::Get(symbol_id)
class SymbolTable {
 public:
  SymbolTable();
  
  // Read synbol table file
  Status Read(const std::string &filename);

  // Get symbol by id
  const char *Get(int symbol_id);

  // Ids for BOS/EOS tag
  int bos_id() const { return bos_id_; }
  int eos_id() const { return eos_id_; }

 private:
  std::vector<std::string> words_;
  std::unordered_map<std::string, int> word_ids_;

  int bos_id_;
  int eos_id_;
};

}  // namespace pocketkaldi

#endif  // POCKETKALDI_SYMBOL_TABLE_H_
