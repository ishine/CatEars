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

void pk_status_init(pk_status_t *status) {
  status->ok = true;
  status->errcode = 0;
  status->message[0] = '\0';
}

void pk_status_fail(pk_status_t *status, int errcode, const char *fmsg, ...) {
  status->ok = false;
  status->errcode = errcode;
  const char *error_prefix = NULL;
  char unknown_error[128];
  char msg[128];

  // Gets error message from fmsg
  va_list args;
  va_start(args, fmsg);
  vsnprintf(msg, sizeof(msg), fmsg, args);
  va_end(args);
  
  // Gets string representation of error code
  switch (errcode) {
    case PK_STATUS_STIOERROR:
      error_prefix = "IOError";
      break;
    case PK_STATUS_STCORRUPTED:
      error_prefix = "Corrupted";
      break;
    default:
      snprintf(
          unknown_error,
          sizeof(unknown_error),
          "UnknownError(%d)",
          errcode);
      error_prefix = unknown_error;
  }

  snprintf(
      status->message,
      PK_STATUS_MSGMAX,
      "%s: %s",
      error_prefix,
      msg);
}

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


ReadableFile::ReadableFile(): fd_(nullptr), file_size_(0) {
}

ReadableFile::~ReadableFile() {
  if (fd_ != nullptr) fclose(fd_);
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
  if (fd_ != nullptr) fclose(fd_);
  fd_ = nullptr;
}

void ToRawStatus(const Status &s, pk_status_t *cs) {
  cs->errcode = s.code();
  cs->ok = s.ok();
  pk_strlcpy(cs->message, s.what().c_str(), PK_STATUS_MSGMAX);
}

}  // namespace util
}  // namespace pocketkaldi
