// Create at 2017-03-27

#include "symbol_table.h"

#include <assert.h>
#include <stdlib.h>
#include "util.h"

namespace pocketkaldi {

SymbolTable::SymbolTable(): size_(0) {}

Status SymbolTable::Read(const std::string &filename) {
  util::ReadableFile fd;
  PK_CHECK_STATUS(fd.Open(filename));
  PK_CHECK_STATUS(fd.ReadAndVerifyString(PK_SYMBOLTABLE_SECTION));

  int32_t section_size = 0;
  PK_CHECK_STATUS(fd.ReadValue<int32_t>(&section_size));

  int32_t table_size = 0;
  int32_t buffer_size = 0;
  PK_CHECK_STATUS(fd.ReadValue<int32_t>(&table_size));
  PK_CHECK_STATUS(fd.ReadValue<int32_t>(&buffer_size));
  size_ = table_size;

  // Check section size
  int expected_size = 8 + table_size * sizeof(int) + buffer_size;
  if (section_size != expected_size) {
    return Status::Corruption(util::Format(
        "pk_symboltable_read: section_size = {} expected, but {} found ({})",
        expected_size,
        section_size,
        filename));
  }

  // Read index
  buffer_index_.resize(table_size);
  int32_t index = 0;
  for (int i = 0; i < table_size; ++i) {
    PK_CHECK_STATUS(fd.ReadValue<int32_t>(&index));
    buffer_index_[i] = index;
  }

  // Read buffer
  buffer_.resize(buffer_size);
  fd.Read(buffer_.data(), buffer_size);

  return Status::OK();
}

const char *SymbolTable::Get(int symbol_id) {
  assert(symbol_id < size_ && "symbol_id out of boundary");
  int idx = buffer_index_[symbol_id];
  assert(idx < buffer_.size() && "symbol index out of boundary");
  return buffer_.data() + idx;
}

}  // namespace pocketkaldi
