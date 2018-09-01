// Create at 2017-03-27

#ifndef POCKETKALDI_SYMBOL_TABLE_H_
#define POCKETKALDI_SYMBOL_TABLE_H_

#include <string>
#include <vector>
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

 private:
  int size_;
  std::vector<char> buffer_;
  std::vector<int> buffer_index_;
};

}  // namespace pocketkaldi

#endif  // POCKETKALDI_SYMBOL_TABLE_H_
