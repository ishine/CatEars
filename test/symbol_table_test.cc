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
  Status status = symbol_table.Read(TESTDIR "data/symboltable_test.bin");
  puts("22");
  assert(status.ok());

  puts("33");
  assert(strcmp(symbol_table.Get(0), "hello") == 0);
  assert(strcmp(symbol_table.Get(1), "world") == 0);
  assert(strcmp(symbol_table.Get(2), "cat") == 0);
  assert(strcmp(symbol_table.Get(3), "milk") == 0);
}

int main() {
  TestSymbolTable();
  return 0;
}
