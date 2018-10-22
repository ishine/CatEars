// Created at 2016-11-24

#include "util.h"

#include <assert.h>
#include <ctype.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <array>
#include <algorithm>

namespace pocketkaldi {
namespace util {

std::string Trim(const std::string &str) {
  std::string::const_iterator begin = str.cbegin();
  std::string::const_iterator end = str.cend() - 1;

  while (begin < str.cend() && isspace(*begin)) ++begin;
  while (end > begin && isspace(*end)) --end;
  return std::string(begin, end + 1);
}

std::vector<std::string> Split(
    const std::string &str,
    const std::string &delim) {
  std::vector<std::string> fields;
  int start = 0;
  int pos = 0;
  while ((pos = str.find(delim, start)) != std::string::npos) {
    fields.emplace_back(str.cbegin() + start, str.cbegin() + pos);
    start = pos + delim.size();
  }
  if (str.cbegin() + start < str.cend()) {
    fields.emplace_back(str.cbegin() + start, str.cend());
  }

  return fields;
}

std::string Tolower(const std::string &str) {
  std::string lower(str.begin(), str.end());
  std::transform(lower.begin(), lower.end(), lower.begin(), tolower);
  return lower;
}


Status StringToLong(const std::string &str, long *val) {
  std::string trim_str = Trim(str);
  char *end = nullptr;
  *val = strtol(trim_str.c_str(), &end, 0);
  if (*end != '\0') {
    return Status::Corruption(Format("unexpected number string: {}", trim_str));
  }
  
  return Status::OK();
}


ReadableFile::ReadableFile(): fd_(nullptr), file_size_(0), owned_(true) {
}

ReadableFile::ReadableFile(FILE *fd):
    fd_(fd), file_size_(0), owned_(false) {
}

ReadableFile::~ReadableFile() {
  if (fd_ != nullptr && owned_) fclose(fd_);
  fd_ = NULL;
}

Status ReadableFile::Open(const std::string &filename) {
  filename_ = filename;
  fd_ = fopen(filename.c_str(), "rb");
  if (fd_ == NULL) {
    return Status::IOError(util::Format("Unable to open {}", filename));
  }

  // Get file size
  fseek(fd_, 0, SEEK_END);
  file_size_ = ftell(fd_);
  fseek(fd_, 0, SEEK_SET);

  return Status::OK();
}

bool ReadableFile::ReadLine(std::string *line, Status *status) {
  // Failed if it already reached EOF
  if (feof(fd_)) {
    *status = Status::IOError(
        util::Format("EOF already reached: {}", filename_));
    return false;
  }

  // Readline
  std::array<char, 4096> chunk;
  char *s = fgets(chunk.data(), chunk.size(), fd_);
  if (s == NULL) {
    if (feof(fd_)) {
      // First time that reached EOF
      return false;
    } else {
      *status = Status::IOError(filename_);
      return false;
    }
  }

  // Trim the tailing '\r' or '\n'
  *line = s;
  while (line->empty() == false &&
         (line->back() == '\r' || line->back() == '\n')) {
    line->pop_back();
  }
  return true;
}

Status ReadableFile::Read(void *ptr, int size) {
  if (1 != fread(ptr, size, 1, fd_)) {
    return Status::IOError(util::Format("failed to read: {}", filename_));
  } else {
    return Status::OK();
  }
}

Status ReadableFile::ReadAndVerifyString(const std::string &expected) {
  std::vector<char> name_buffer(expected.size() + 1);

  Status status = Read(name_buffer.data(), expected.size());
  if (!status.ok()) return status;
  name_buffer.back() = '\0';
  if (expected != name_buffer.data()) {
    return Status::Corruption(util::Format(
       "ReadAndVerifyString: '{}' expected but '{}' found in {}",
       expected,
       name_buffer.data(),
       filename_));
  }

  return Status::OK();
}

bool ReadableFile::Eof() const {
  return feof(fd_) != 0;
}

void ReadableFile::Close() {
  assert(owned_ && "unable to call Close() in borrowed mode");
  if (fd_ != nullptr) fclose(fd_);
  fd_ = nullptr;
}

}  // namespace util
}  // namespace pocketkaldi
