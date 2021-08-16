/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_LIB_IO_PROGRESSIVE_COMPRESSED_RECORD_READER_H_
#define TENSORFLOW_CORE_LIB_IO_PROGRESSIVE_COMPRESSED_RECORD_READER_H_

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

struct ProgressiveCompressedRecordReaderOptions {
  enum CompressionType {
    NONE = 0,
  };
  CompressionType compression_type = NONE;

  // If buffer_size is non-zero, then all reads must be sequential, and no
  // skipping around is permitted. (Note: this is the same behavior as reading
  // compressed files.) Consider using SequentialRecordReader.
  int64 buffer_size = 0;

  static ProgressiveCompressedRecordReaderOptions CreateProgressiveCompressedRecordReaderOptions(
      const string& compression_type);

};

// Low-level interface to read TFRecord files.
//
// If using compression or buffering, consider using SequentialRecordReader.
//
// Note: this class is not thread safe; external synchronization required.
class ProgressiveCompressedRecordReader {
 public:

  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  explicit ProgressiveCompressedRecordReader(
      RandomAccessFile* file,
      const std::vector<int32>& record_offsets,
      const ProgressiveCompressedRecordReaderOptions& options = ProgressiveCompressedRecordReaderOptions());

  virtual ~ProgressiveCompressedRecordReader() = default;

  // Read the record up to scan into *record and update *offset to
  // point to the offset of the next record.  Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(int32 scan, tstring* record);

 private:
  Status Read(size_t n, tstring* result);

  ProgressiveCompressedRecordReaderOptions options_;
  std::unique_ptr<InputStreamInterface> input_stream_;
  bool last_read_failed_;
  std::vector<int32> record_offsets_;

  TF_DISALLOW_COPY_AND_ASSIGN(ProgressiveCompressedRecordReader);
};

// High-level interface to read ProgressiveCompressedRecord files.
//
// Note: this class is not thread safe; external synchronization required.
class SequentialProgressiveCompressedRecordReader {
 public:
  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  explicit SequentialProgressiveCompressedRecordReader(
      RandomAccessFile* file,
      const std::vector<int32>& record_offsets,
      const ProgressiveCompressedRecordReaderOptions& options = ProgressiveCompressedRecordReaderOptions());

  virtual ~SequentialProgressiveCompressedRecordReader() = default;

  // Read the next record in the file into *record. Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(int32 scan, tstring* record) {
    return underlying_.ReadRecord(scan, record);
  }

 private:
  ProgressiveCompressedRecordReader underlying_;
};


}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_
