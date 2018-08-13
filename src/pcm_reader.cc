// 2017-01-23

#include "pcm_reader.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "util.h"

// Reads an int32 value from ptr and move it forward
static int32_t read_int32(char **ptr) {
  int32_t val = *((int32_t *)*ptr);
  *ptr += sizeof(int32_t);
  return val;
}

// Reads an int16 value from ptr and move it forward
static int16_t read_int16(char **ptr) {
  int16_t val = *((int16_t *)*ptr);
  *ptr += sizeof(int16_t);
  return val;
}

// Reads an int8 value from ptr and move it forward
static int8_t read_int8(char **ptr) {
  int8_t val = *((int8_t *)*ptr);
  *ptr += sizeof(int8_t);
  return val;
}

// Reads a string with length strlen(expected) from ptr, and compared with
// expected. If the same returns true, otherwise returns false. Then move ptr
// forward strlen(expected) bytes
static bool check_tag(char **ptr, const char *expected) {
  int len = strlen(expected);
  bool is_same = strncmp(*ptr, expected, len) == 0;
  *ptr += len;
  return is_same;
}

namespace pocketkaldi {

// Reads 16k sampling rate, mono-channel, PCM formatted wave file, and stores
// the data into data. If any error occured, set status to failed
Status Read16kPcm(const char *filename, Vector<float> *pcm_data) {
  Status status;
  util::ReadableFile fd;
  status = fd.Open(filename);


  // Gets file length
  int64_t file_size = 0;
  if (status.ok()) {
    file_size = fd.file_size();
  }

  // Read file content into pcm_buffer
  char *current_ptr = NULL;
  std::vector<char> pcm_buffer;
  if (status.ok()) {
    pcm_buffer.resize(file_size);
    current_ptr = pcm_buffer.data();
    status = fd.Read(pcm_buffer.data(), pcm_buffer.size());
  }

  // RIFF chunk
  if (status.ok() && check_tag(&current_ptr, "RIFF") == false) {
    return Status::Corruption(util::Format(
        "chunk_name == 'RIFF' expected: {}",
        filename));
  }

  // Chunk size
  if (status.ok()) {
    int chunk_size = read_int32(&current_ptr);
    if (chunk_size != file_size - 8) {
      return Status::Corruption(util::Format(
          "chunk_size == {} expected, but {} found: {}",
          file_size - 8,
          chunk_size,
          filename));
    }
  }

  // Format == "WAVE"
  if (status.ok() && check_tag(&current_ptr, "WAVE") == false) {
    return Status::Corruption(util::Format(
        "Format == 'WAVE' expected: {}",
        filename));
  }

  // subchunk1 is "fmt "
  if (status.ok() && check_tag(&current_ptr, "fmt ") == false) {
    return Status::Corruption(util::Format(
        "subchunk1 == 'fmt ' expected: {}",
        filename));
  }  

  // subchunk1_size
  if (status.ok()) {
    int subchunk1_size = read_int32(&current_ptr);
    if (subchunk1_size != 16) {
      return Status::Corruption(util::Format(
          "subchunk1_size == 16 expected, but {} found: {}",
          subchunk1_size,
          filename));
    }
  }

  // audio_format
  if (status.ok()) {
    int audio_format = read_int16(&current_ptr);
    if (audio_format != 1) {
      return Status::Corruption(util::Format(
          "audio_format == 1 (PCM) expected, but {} found: {}",
          audio_format,
          filename));
    }
  }

  // num_channels
  if (status.ok()) {
    int num_channels = read_int16(&current_ptr);
    if (num_channels != 1) {
      return Status::Corruption(util::Format(
          "num_channels == 1 (mono) expected, but {} found: {}",
          num_channels,
          filename));
    }
  }

  // sample_rate
  int sample_rate = 0;
  if (status.ok()) {
    sample_rate = read_int32(&current_ptr);
    if (sample_rate != 16000) {
      return Status::Corruption(util::Format(
          "sample_rate == 16000 expected, but {} found: {}",
          sample_rate,
          filename));
    }
  }

  // bytes_rate, block_align, bits_per_sample
  int bits_per_sample = 0;
  if (status.ok()) {
    int bytes_rate = read_int32(&current_ptr);
    int block_align = read_int16(&current_ptr);
    bits_per_sample = read_int16(&current_ptr);

    if (bytes_rate != sample_rate * bits_per_sample / 8) {
      return Status::Corruption(util::Format(
          "bytes_rate == {} expected, but {} found: {}",
          sample_rate * bits_per_sample / 8,
          bytes_rate,
          filename));
    }

    if (block_align != bits_per_sample / 8) {
      return Status::Corruption(util::Format(
          "block_align == {} expected, but {} found: {}",
          bits_per_sample / 8,
          bytes_rate,
          filename));
    }
  }

  // subchunk2 "data"
  if (status.ok() && check_tag(&current_ptr, "data") == false) {
    return Status::Corruption(util::Format(
        "subchunk2 == 'data' expected: {}",
        filename));
  }

  // subchunk2_size
  int64_t subchunk2_size = 0;
  if (status.ok()) {
    subchunk2_size = read_int32(&current_ptr);
    if (subchunk2_size != file_size - 44) {
      return Status::Corruption(util::Format(
          "subchunk2_size == {} expected, but {} found: {}",
          file_size - 44,
          subchunk2_size,
          filename));
    }
  }

  // Read data
  if (status.ok()) {
    int num_samples = (int)subchunk2_size / (bits_per_sample / 8);
    pcm_data->Resize(num_samples);
    for (int i = 0; i < num_samples && status.ok(); ++i) {
      switch (bits_per_sample) {
        case 8:
          (*pcm_data)(i) = read_int8(&current_ptr);
          break;
        case 16:
          (*pcm_data)(i) = read_int16(&current_ptr);
          break;
        case 32:
          (*pcm_data)(i) = read_int32(&current_ptr);
          break;
        default:
          return Status::Corruption(util::Format(
              "bits_per_sample == 8, 16 or 32 expected, but {} found: {}",
              bits_per_sample,
              filename));
      }
    }
  }

  if (status.ok()) {
    if (current_ptr != pcm_buffer.data() + file_size) {
      return Status::Corruption(util::Format(
          "unexpected file size: {}",
          filename));
    }
  }

  return Status::OK();
}

}  // namespace pocketkaldi
