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

#include "tensorflow/core/lib/io/progressive_compressed_record_reader.h"

#include <limits.h>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {

ProgressiveCompressedRecordReaderOptions ProgressiveCompressedRecordReaderOptions::CreateProgressiveCompressedRecordReaderOptions(
    const string& compression_type) {
  ProgressiveCompressedRecordReaderOptions options;

  if (compression_type != compression::kNone) {
    LOG(ERROR) << "Unsupported compression_type:" << compression_type
               << ". No compression will be used.";
  }
  return options;
}

ProgressiveCompressedRecordReader::ProgressiveCompressedRecordReader(
    RandomAccessFile* file,
    const std::vector<int32>& record_offsets,
    const ProgressiveCompressedRecordReaderOptions& options)
    : options_(options),
      input_stream_(new RandomAccessInputStream(file)),
      last_read_failed_(false),
      record_offsets_(record_offsets) {
  if (options.buffer_size > 0) {
    input_stream_.reset(new BufferedInputStream(input_stream_.release(),
                                                options.buffer_size, true));
  }
  if (options.compression_type == ProgressiveCompressedRecordReaderOptions::NONE) {
    // Nothing to do.
  } else {
    LOG(FATAL) << "Unrecognized compression type :" << options.compression_type;
  }
}

// Read n bytes
Status ProgressiveCompressedRecordReader::Read(size_t n, tstring* result) {
  if (n >= SIZE_MAX) {
    return errors::DataLoss("record size too large");
  }

  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(n, result));

  if (result->size() != n) {
    if (result->empty()) {
      return errors::OutOfRange("eof");
    }
  }

  result->resize(n); // TODO(mkuchnik): Remove?
  return Status::OK();
}

Status ProgressiveCompressedRecordReader::ReadRecord(int32 scan, tstring* record) {
  const uint64 length = record_offsets_.at(scan);

  // Read data
  Status s = Read(length, record);
  if (!s.ok()) {
    last_read_failed_ = true;
    if (errors::IsOutOfRange(s)) {
      s = errors::DataLoss("truncated record failed with ",
                           s.error_message());
    }
    return s;
  }

  return Status::OK();
}

SequentialProgressiveCompressedRecordReader::SequentialProgressiveCompressedRecordReader(
    RandomAccessFile* file,
    const std::vector<int32>& record_offsets,
    const ProgressiveCompressedRecordReaderOptions& options)
    : underlying_(file, record_offsets, options) {}

}  // namespace io
}  // namespace tensorflow
