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

#include "tensorflow/core/lib/io/progressive_compressed_record_writer.h"

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/example/progressive_compressed_record.pb.h"

namespace tensorflow {
namespace io {

PCRWriterOptions PCRWriterOptions::CreateRecordWriterOptions(
    const string& compression_type) {
  PCRWriterOptions options;
  if (compression_type != compression::kNone) {
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
  }
  return options;
}

PCRWriter::PCRWriter(WritableFile* dest,
                     const PCRWriterOptions& options)
    : dest_(dest), options_(options) {
  if (options.compression_type != PCRWriterOptions::NONE) {
    LOG(FATAL) << "Compression is unsupported on mobile platforms.";
  }
}

PCRWriter::~PCRWriter() {
  if (dest_ != nullptr) {
    Status s = Close();
    if (!s.ok()) {
      LOG(ERROR) << "Could not finish writing file: " << s;
    }
  }
}

int64 find_largest_scan(const std::vector< std::vector<StringPiece> >& data) {
  auto vector_size_compare = [](
      const std::vector<StringPiece>& lhs,
      const std::vector<StringPiece>& rhs) { return lhs.size() < rhs.size(); };
  const auto largest_scan = std::max_element(
      data.begin(),
      data.end(),
      vector_size_compare
  )->size();
  return largest_scan;
}

Status PCRWriter::WriteRecords(
    const std::vector<int32>& labels,
    const std::vector< std::vector<StringPiece> >& data,
    const std::vector<Example>& metadata,
    std::vector<size_t>& offsets
    ) {
  if (dest_ == nullptr) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Writer not initialized or previously closed");
  }
  if (offsets.size()) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Offsets is not empty");
  }
  // First we find max size to simplify cases. All vecs should have same size.
  const auto max_size = std::max(
      {labels.size(), data.size(), metadata.size()});
  if (labels.size() != max_size ||
      data.size()  != max_size ||
      metadata.size() != max_size) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "The size of labels, data, and metadata must match.");
  }
  const auto largest_scan = find_largest_scan(data);

   // For storing indexing information
  // Next, we write out metadata. Labels and free-form metadata are embedded
  // first.
  MetadataRecord metarecord;
  *metarecord.mutable_labels() = {labels.begin(), labels.end()};
  metarecord.set_progressive_levels(largest_scan);
  metarecord.set_version(100);  // arbitrary
  *metarecord.mutable_examples() = {metadata.begin(), metadata.end()};
  std::string payload;
  if (!metarecord.SerializeToString(&payload)) {
    LOG(ERROR) << "Could not serialize metadata";
    return Status(::tensorflow::error::INTERNAL,
                  "Could not serialize metadata.");
  }
  TF_RETURN_IF_ERROR(dest_->Append(payload));
  offsets.push_back(payload.size());

  // Finally, we write out data. First scan is written first, followed by other
  // scans
  payload.clear();
  ScanGroup scangroup;
  for (size_t scan = 0; scan < largest_scan; scan++) {
    std::vector<std::string> curr_scans;
    for (const auto inner_vec : data) {
      curr_scans.push_back(std::string(inner_vec.at(scan)));
    }
    *scangroup.mutable_image_bytes() = {curr_scans.begin(), curr_scans.end()};
    if (!scangroup.SerializeToString(&payload)) {
      LOG(ERROR) << "Could not serialize data";
      return Status(::tensorflow::error::INTERNAL,
                    "Could not serialize data.");
    }
    TF_RETURN_IF_ERROR(dest_->Append(payload));
    offsets.push_back(payload.size());
    payload.clear();
  }

  return Status::OK();
}

Status PCRWriter::Close() {
  return Status::OK();
}

Status PCRWriter::Flush() {
  if (dest_ == nullptr) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Writer not initialized or previously closed");
  }
  return dest_->Flush();
}

}  // namespace io
}  // namespace tensorflow
