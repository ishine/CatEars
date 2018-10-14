// 2017-01-23

#include "pcm_reader.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "util.h"

namespace {

// Reads an int32 value from ptr and move it forward
static int32_t ReadInt32(char **ptr) {
  int32_t val = *((int32_t *)*ptr);
  *ptr += sizeof(int32_t);
  return val;
}

// Reads an int16 value from ptr and move it forward
static int16_t ReadInt16(char **ptr) {
  int16_t val = *((int16_t *)*ptr);
  *ptr += sizeof(int16_t);
  return val;
}

// Reads an int8 value from ptr and move it forward
static int8_t ReadInt8(char **ptr) {
  int8_t val = *((int8_t *)*ptr);
  *ptr += sizeof(int8_t);
  return val;
}


}  // namespace

namespace pocketkaldi {

Status ReadPcmHeader(util::ReadableFile *fd, pasco_wave_format_t *wave_fmt) {
  PK_CHECK_STATUS(fd->ReadAndVerifyString("RIFF"));

  int32_t chunk_size = 0;
  PK_CHECK_STATUS(fd->ReadValue<int32_t>(&chunk_size));
  if (chunk_size != fd->file_size() - 8) {
    return Status::Corruption(util::Format(
        "chunk_size == {} expected, but {} found: {}",
        fd->file_size() - 8,
        chunk_size,
        fd->filename()));
  }

  // Format == "WAVE"
  PK_CHECK_STATUS(fd->ReadAndVerifyString("WAVE"));
 
  // Subchunk1
  PK_CHECK_STATUS(fd->ReadAndVerifyString("fmt "));
  int32_t subchunk1_size = 0;
  PK_CHECK_STATUS(fd->ReadValue(&subchunk1_size));
  if (subchunk1_size != 16) {
    return Status::Corruption(util::Format(
        "subchunk1_size == 16 expected, but {} found: {}",
        subchunk1_size,
        fd->filename()));
  }

  // audio_format
  int16_t audio_format = 0;
  PK_CHECK_STATUS(fd->ReadValue(&audio_format));
  if (audio_format != 1) {
    return Status::Corruption(util::Format(
        "audio_format == 1 expected, but {} found: {}",
        audio_format,
        fd->filename()));
  }

  // num_channels
  int16_t num_channels = 0;
  PK_CHECK_STATUS(fd->ReadValue(&num_channels));
  wave_fmt->num_channels = num_channels;

  int32_t sample_rate = 0;
  PK_CHECK_STATUS(fd->ReadValue(&sample_rate));
  wave_fmt->sample_rate = sample_rate;

  // bytes_rate, block_align, bits_per_sample
  int16_t bits_per_sample = 0;
  int32_t bytes_rate = 0;
  int16_t block_align = 0;
  PK_CHECK_STATUS(fd->ReadValue(&bytes_rate));
  PK_CHECK_STATUS(fd->ReadValue(&block_align));
  PK_CHECK_STATUS(fd->ReadValue(&bits_per_sample));

  if (bytes_rate != sample_rate * bits_per_sample / 8) {
    return Status::Corruption(util::Format(
        "bytes_rate == {} expected, but {} found: {}",
        sample_rate * bits_per_sample / 8,
        bytes_rate,
        fd->filename()));
  }

  if (block_align != bits_per_sample / 8) {
    return Status::Corruption(util::Format(
        "block_align == {} expected, but {} found: {}",
        bits_per_sample / 8,
        bytes_rate,
        fd->filename()));
  }
  wave_fmt->bits_per_sample = bits_per_sample;

  // subchunk2 "data"
  PK_CHECK_STATUS(fd->ReadAndVerifyString("data"));

  // subchunk2_size
  int32_t subchunk2_size = 0;
  PK_CHECK_STATUS(fd->ReadValue(&subchunk2_size));
  if (subchunk2_size != fd->file_size() - 44) {
    return Status::Corruption(util::Format(
        "subchunk2_size == {} expected, but {} found: {}",
        fd->file_size() - 44,
        subchunk2_size,
        fd->filename()));
  }

  return Status::OK();
}

WaveReader::WaveReader(): ready_(false) {}

Status WaveReader::SetFormat(const pasco_wave_format_t &format) {
  // num_channels
  if (format.num_channels != 1) {
    return Status::Corruption(util::Format(
        "num_channels = {} not supported",
        format.num_channels));
  }

  // sample_rate
  if (format.sample_rate != 16000) {
    return Status::Corruption(util::Format(
        "sample_rate = {} not supported",
        format.sample_rate));
  }

  // bits_per_sample
  switch (format.bits_per_sample) {
    case 8:
    case 16:
    case 32:
      break;
    default:
      return Status::Corruption(util::Format(
          "bits_per_sample == 8, 16 or 32 expected, but {} found",
          format.bits_per_sample));
  }

  ready_ = true;
  format_ = format;
  return Status::OK();
}

Status WaveReader::Process(
    const char *buffer, int size, Vector<float> *pcm_data) {
  if (!buffer) {
    return Status::RuntimeError("buffer is nullptr");
  }
  if (size <= 0) {
    return Status::RuntimeError(util::Format("unexpected size: {}", size));
  }
  if (!ready_) {
    return Status::RuntimeError("WaveReader is not ready");
  }

  // Update buffer
  buffer_.insert(buffer_.end(), buffer, buffer + size);

  // Compute bytes to process
  int bytes_per_sample = format_.bits_per_sample / 8;
  int num_samples = buffer_.size() / bytes_per_sample;
  pcm_data->Resize(num_samples);
  char *current_ptr = buffer_.data();
  for (int i = 0; i < num_samples; ++i) {
    switch (format_.bits_per_sample) {
      case 8:
        (*pcm_data)(i) = ReadInt8(&current_ptr);
        break;
      case 16:
        (*pcm_data)(i) = ReadInt16(&current_ptr);
        break;
      case 32:
        (*pcm_data)(i) = ReadInt32(&current_ptr);
        break;
      default:
        assert(false && "unexpected bits_per_sample");
    }
  }

  // Update buffer
  int processed_bytes = bytes_per_sample * num_samples;
  assert(processed_bytes <= buffer_.size());
  buffer_.erase(buffer_.begin(), buffer_.begin() + processed_bytes);

  return Status::OK();
}

// Reads 16k sampling rate, mono-channel, PCM formatted wave file, and stores
// the data into data. If any error occured, set status to failed
Status Read16kPcm(const char *filename, Vector<float> *pcm_data) {
  pasco_wave_format_t fmt;
  
  util::ReadableFile fd;
  PK_CHECK_STATUS(fd.Open(filename));

  PK_CHECK_STATUS(ReadPcmHeader(&fd, &fmt));
  WaveReader reader;
  PK_CHECK_STATUS(reader.SetFormat(fmt));

  int data_size = static_cast<int>(fd.file_size() - 44);
  std::vector<char> buffer(data_size);
  PK_CHECK_STATUS(fd.Read(buffer.data(), data_size));

  PK_CHECK_STATUS(reader.Process(buffer.data(), data_size, pcm_data));
  
  return Status::OK();
}

}  // namespace pocketkaldi
