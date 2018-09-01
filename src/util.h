// Created at 2016-11-24

#ifndef POCKETKALDI_UTIL_H_
#define POCKETKALDI_UTIL_H_

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>
#include <utility>
#include <vector>
#include "pocketkaldi.h"
#include "status.h"

// Error types for status
#define PK_STATUS_STSUCCESS 0
#define PK_STATUS_STIOERROR 1
#define PK_STATUS_STCORRUPTED 2


#define PK_UNUSED(x) (void)(x)
#define PK_MIN(a, b) ((a) < (b) ? (a) : (b))

#define PK_PATHMAX 1024

#define PK_CHECK_STATUS(st_exp) {\
    Status st = (st_exp);\
    if (!st.ok()) return st;}

#define PK_INFO(msg) std::cout << __FILE__ << ": " << (msg) << std::endl;
#define PK_WARN(msg) std::cout << "WARN: " << __FILE__ << ": " \
                               << (msg) << std::endl;
// #define PK_DEBUG(msg) std::cout << __FILE__ << ": " << (msg) << std::endl;
#define PK_DEBUG(msg)

// Initialize the status set to success (ok) state
POCKETKALDI_EXPORT
void pk_status_init(pk_status_t *status);

// Set status to failed state with message
POCKETKALDI_EXPORT
void pk_status_fail(pk_status_t *status, int errcode, const char *fmsg, ...);

#define PK_STATUS_IOERROR(status, fmsg, ...) \
    pk_status_fail(status, PK_STATUS_STIOERROR, fmsg, __VA_ARGS__)

#define PK_STATUS_CORRUPTED(status, fmsg, ...) \
    pk_status_fail(status, PK_STATUS_STCORRUPTED, fmsg, __VA_ARGS__)

// The same as strlcpy in FreeBSD
size_t pk_strlcpy(char *dst, const char *src, size_t siz);

// 
namespace pocketkaldi {
namespace util {

template<typename T,
         typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline std::string ToString(const T &val) {
  return std::to_string(val);
}
template<typename T>
inline std::string ToString(const T * const val) {
  return std::to_string(reinterpret_cast<uint64_t>(val));
}
inline std::string ToString(const std::string &val) {
  return val;
}
inline std::string ToString(const char *const &val) {
  return std::string(val);
}

namespace {

inline std::string FormatImpl(const std::string &formatted) {
  return formatted;
}
template<typename T, typename... Args>
inline std::string FormatImpl(
    const std::string &formatted,
    T &&item,
    Args &&...args) {
  size_t pos = formatted.find("{}");
  std::string repl = ToString(item);
  std::string next_formatted = formatted;
  if (pos != std::string::npos) {
    next_formatted.replace(pos, 2, repl);
  }
  return FormatImpl(next_formatted, std::forward<Args>(args)...);
}

}  // namespace

// Format string function, just like Python, it uses '{}' for replacement. For
// example:
//   util::Format("Hello, {}, {}!", "World", "2233");
template<typename... Args>
inline std::string Format(const std::string &fmt, Args &&...args) {
  return FormatImpl(fmt, std::forward<Args>(args)...);
}

// Trim string 
std::string Trim(const std::string &str);

// Split string by delim and returns as a vector of strings
std::vector<std::string> Split(
    const std::string &str,
    const std::string &delim);

// String tolower
std::string Tolower(const std::string &str);

// A wrapper of FILE in stdio.h
class ReadableFile {
 public:
  ReadableFile();
  ~ReadableFile();

  // Return true if end-of-file reached
  bool Eof() const;

  // Open a file for read. If success, status->ok() will be true. Otherwise,
  // status->ok() == false
  Status Open(const std::string &filename);

  // Read n bytes (size) from file and put to *ptr
  Status Read(void *ptr, int size);

  // Read a string with size `expected.size()` from file. Then compare it with
  // `expected`. If different, return a failed state. Otherwise, return success
  Status ReadAndVerifyString(const std::string &expected);

  // Read an type T from file
  template<typename T>
  Status ReadValue(T *data) {
    return Read(data, sizeof(T));
  }

  // Get filename
  const std::string &filename() const {
    return filename_;
  }

  // Return filesize
  int64_t file_size() const {
    return file_size_;
  }

  // Close opened file
  void Close();

  // Read a line from file. 
  //     On success: status->ok() == true and return true.
  //     On EOF reached first time: status->ok() == true and return false.
  //     Other: status->ok() == false and return false.
  bool ReadLine(std::string *line, Status *status);

 private:
  std::string filename_;
  FILE *fd_;
  int64_t file_size_;
};

// To check if a class have 'previous' field
template <typename T>
struct has_previous {
  template<typename C> static int8_t check(decltype(&C::previous)) ;
  template<typename C> static int16_t check(...);    
  enum {
    value = (sizeof(check<T>(0)) == sizeof(int8_t))
  };
};

// Convert to C status
void ToRawStatus(const Status &s, pk_status_t *cs);

}  // namespace util
}  // namespace pocketkaldi

#endif  // POCKETKALDI_UTIL_H_
