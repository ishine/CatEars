// Created at 2017-04-01

#include "symbol_table.h"

#include <assert.h>
#include <string.h>
#include "pocketkaldi.h"
#include "util.h"

using pocketkaldi::SymbolTable;
using pocketkaldi::Status;

void TestSymbolTable() {
  SymbolTable symbol_table;
  Status status = symbol_table.Read(TESTDIR "data/lm.words.txt");
  assert(status.ok());


  assert(strcmp(symbol_table.Get(958), "marisa") == 0);
  assert(strcmp(symbol_table.Get(1272), "reimu") == 0);
  assert(strcmp(symbol_table.Get(1839), "zun") == 0);
  assert(strcmp(symbol_table.Get(0), "<eps>") == 0);

  assert(symbol_table.bos_id() == 2);
  assert(symbol_table.eos_id() == 1);
}

int main() {
  TestSymbolTable();
  return 0;
}
