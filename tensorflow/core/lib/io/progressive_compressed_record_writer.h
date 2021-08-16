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

#ifndef TENSORFLOW_CORE_LIB_IO_PROGRESSIVE_COMPRESSED_RECORD_WRITER_H_
#define TENSORFLOW_CORE_LIB_IO_PROGRESSIVE_COMPRESSED_RECORD_WRITER_H_

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/example/example.pb.h"

namespace tensorflow {

class WritableFile;

namespace io {

struct PCRWriterOptions {
 public:
  enum CompressionType {
    NONE = 0
  };
  CompressionType compression_type = NONE;

  static PCRWriterOptions CreateRecordWriterOptions(
      const string& compression_type);
};

class PCRWriter {
 public:
  // Create a writer that will append data to "*dest".
  // "*dest" must be initially empty.
  // "*dest" must remain live while this Writer is in use.
  PCRWriter(WritableFile* dest,
            const PCRWriterOptions& options = PCRWriterOptions());

  // Calls Close() and logs if an error occurs.
  ~PCRWriter();

  // Takes a collection of labels, data, and metadata and writes them out to a
  // PCR file. The size of each of these vectors should be the same if they are
  // not null. PCR format dictates that labels and metadata will be written out
  // before the data (usually image data). Data will be written out in
  // transposed format, where the data layout will have inner data before outer
  // data.
  // TODO(mkuchnik): Null records not yet supported
  //
  // On disk example:
  // labels: 0, 1, 5, 0, 1, ...
  // metadata: example1, example2, example3, ...
  // data group 1: data1_1, data1_2, data1_3, ...
  // data group 2: data2_1, data2_2, data2_3, ...
  // ...
  // data group N: dataN_1, dataN_2, dataN_3, ...
  // TODO(mkuchnik): The functionality here can be moved into a variable length
  // record combined with a fixed/tfrecord format for metadata.
  //
  // Offsets for the record are written to 'offsets'
  // these can then be written using a different record writer.
  Status WriteRecords(const std::vector<int32>& labels,
                      const std::vector< std::vector<StringPiece> >& data,
                      const std::vector<Example>& metadata,
                      std::vector<size_t>& offsets
                      );

  // Flushes any buffered data held by underlying containers of the
  // PCRWriter to the WritableFile. Does *not* flush the
  // WritableFile.
  Status Flush();

  // Writes all output to the file. Does *not* close the WritableFile.
  //
  // After calling Close(), any further calls to `WriteRecord()` or `Flush()`
  // are invalid.
  Status Close();

 private:
  WritableFile* dest_;
  PCRWriterOptions options_;

  TF_DISALLOW_COPY_AND_ASSIGN(PCRWriter);
};


}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_PROGRESSIVE_COMPRESSED_RECORD_WRITER_H_
