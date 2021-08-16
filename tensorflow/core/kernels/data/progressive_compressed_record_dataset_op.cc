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
#include "tensorflow/core/kernels/data/progressive_compressed_record_dataset_op.h"

#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/progressive_compressed_record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include <stdlib.h>
#include <malloc.h>
#include <fstream>

#include "tensorflow/core/util/pcr_parsing.h"

namespace tensorflow {
namespace data {


// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following ops.

/* static */ constexpr const char* const ProgressiveCompressedRecordDatasetOp::kDatasetType;
/* static */ constexpr const char* const ProgressiveCompressedRecordDatasetOp::kFileNames;
/* static */ constexpr const char* const ProgressiveCompressedRecordDatasetOp::kCompressionType;
/* static */ constexpr const char* const ProgressiveCompressedRecordDatasetOp::kBufferSize;
/* static */ constexpr const char* const ProgressiveCompressedRecordDatasetOp::kScanGroups;
/* static */ constexpr const char* const ProgressiveCompressedRecordDatasetOp::kIndexSourceFilename;
/* static */ constexpr const char* const ProgressiveCompressedRecordDatasetOp::kMetadataOutputType;

namespace PCR {
constexpr char kCurrentFileIndex[] = "current_file_index";
constexpr char kCurrentRecordIndex[] = "current_record_index";

} // PCR


template<typename T>
static std::vector<T> cumsum(const std::vector<T>& records) {
  std::vector<T> clean_record_offsets(records);
  T cumsum = 0;
  for (size_t i = 0; i < records.size(); i++) {
    cumsum += records.at(i);
    clean_record_offsets.at(i) = cumsum;
  }
  return clean_record_offsets;
}

static bool validate_metadata_output_type(const tstring& metadata_output_type) {
  return metadata_output_type == "" || metadata_output_type == "labels_first";
}

template<typename T>
class IndexCache {
  public:
    void insert(const std::string& key, T item) {
      mutex_lock l(mu_);
      lookup_table.insert( std::pair<std::string, T >(key, item) );
    }
    bool lookup(const std::string& key, T& item) {
      mutex_lock l(mu_);
      const auto& it = lookup_table.find(key);
      if (it == lookup_table.end()) {
        return false;
      }
      item = it->second;
      return true;
    }
  private:
    std::map<std::string, T> lookup_table;
    mutex mu_;
};

template<typename T>
class AllocatorCache {
  public:
    AllocatorCache() : free_vec(16, true), resource_vec(16) {}
    bool request(T& item) {
      mutex_lock l(mu_);
      auto offset = lookup_free();
      if (offset < 0) {
        std::cerr << "Dynamic allocation not implemented" << std::endl;
        return false;
      }
      mark_empty(offset);
      std::swap(item, resource_vec.at(offset));
      VLOG(3) << "Requested";
      return true;
    }
    bool release(T& item) {
      mutex_lock l(mu_);
      auto offset = lookup_empty();
      if (offset < 0) {
        std::cerr << "Dynamic allocation not implemented" << std::endl;
        return false;
      }
      mark_free(offset);
      std::swap(item, resource_vec.at(offset));
      VLOG(3) << "Released";
      return true;
    }
  private:
    int lookup_free() {
      return lookup_val(true);
    }
    int lookup_empty() {
      return lookup_val(false);
    }
    void mark_free(int offset) {
      VLOG(3) << "Free slot " << offset << "/" << free_vec.size();
      free_vec.at(offset) = true;
    }
    void mark_empty(int offset) {
      VLOG(3) << "Taken slot " << offset << "/" << free_vec.size();
      free_vec.at(offset) = false;
    }
    int lookup_val(bool val) {
      const auto& it = std::find(free_vec.begin(), free_vec.end(), val);
      if (it == free_vec.end()) {
        return -1;
      }
      int index = std::distance(free_vec.begin(), it);
      return index;
    }

    // Allocate 32 max
    std::vector<bool> free_vec;
    std::vector<T> resource_vec;
    mutex mu_;
};

void read_index_file(OpKernelContext* ctx,
    const std::string& index_source_filename_,
    std::string& file_content) {
  std::unique_ptr<RandomAccessFile> index_file;
  Status s = ctx->env()->NewRandomAccessFile(index_source_filename_,
                                             &index_file);
  if (!s.ok()) {
    std::cerr << "Error opening file: " << index_source_filename_ << std::endl;
  }
  io::RandomAccessInputStream input_stream(index_file.get());
  while (true) {
    tstring result;
    Status s = input_stream.ReadNBytes(1024*1024, &result);
    file_content += result;
    if (!s.ok()) {
      if (errors::IsOutOfRange(s)) {
        break;
      }
      std::cerr << "Error reading file: " << index_source_filename_ << std::endl;
    }
  }
}


void read_index_file_os(const std::string& index_source_filename_,
                               std::string& file_content) {
  // NOTE(mkuchnik): Untested code for avoiding TF files
  std::ifstream t(index_source_filename_);
  t.seekg(0, std::ios::end);
  size_t size = t.tellg();
  file_content.reserve(size);
  t.seekg(0);
  t.read(&file_content[0], size);
}

static IndexCache< std::map< std::string, std::vector<int32> > > index_cache;
static AllocatorCache< progressive_compressed_record::PCR_State > allocator_cache;

class ProgressiveCompressedRecordDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, std::vector<string> filenames,
                   const string& compression_type, int64 buffer_size, int32 scan_groups,
                   const string& index_source_filename,
                   const string& metadata_output_type,
                   std::vector< std::vector<int32> > record_offsets
                   )
      : DatasetBase(DatasetContext(ctx)),
        filenames_(std::move(filenames)),
        compression_type_(compression_type),
        options_(io::ProgressiveCompressedRecordReaderOptions::CreateProgressiveCompressedRecordReaderOptions(
            compression_type)),
        scan_groups_(scan_groups),
        index_source_filename_(index_source_filename),
        metadata_output_type_(metadata_output_type),
        record_offsets_(record_offsets) {
    if (buffer_size > 0) {
      options_.buffer_size = buffer_size;
    }
    VLOG(3) << "metadata_output_type: " << metadata_output_type_;
    if (index_source_filename_.size() && !record_offsets_.size()) {
      VLOG(ERROR) << "Both index and record_offsets are set";
    }
    if (index_source_filename_.size() && !record_offsets_.size()) {
      VLOG(3) << "Found index_source_filename";
      //std::cout << "Found index_source_filename" << std::endl;
      std::map< std::string, std::vector<int32> > pcr_index;
      if (!index_cache.lookup(index_source_filename_, pcr_index)) {
        VLOG(3) << "Cache miss";
        std::cout << "Cache miss" << std::endl;
        // Cache miss
        std::string index_source_filename_content;
        read_index_file(ctx,
                        index_source_filename_,
                        index_source_filename_content);
        pcr_index = progressive_compressed_record::parse_PCR_index(
            index_source_filename_content);
        index_cache.insert(index_source_filename_, pcr_index);
      }
      record_offsets_.reserve(filenames_.size());
      for (const auto& f: filenames_) {
        const auto slash_it = f.find_last_of("/\\");
        auto it = pcr_index.end();
        if (slash_it != string::npos) {
          const auto base_filename = f.substr(slash_it + 1); // strip path
          it = pcr_index.find(base_filename);
          if (it == pcr_index.end()) {
            // If resolve failed, fall back to full path
            it = pcr_index.find(f);
          }
        } else {
          it = pcr_index.find(f);
        }
        if (it == pcr_index.end()) {
          VLOG(3) << "Missing PCR metadata for " << f;
          std::cerr << "Missing PCR metadata for " << f << std::endl;
          for (const auto& f: pcr_index) {
            VLOG(4) << "key " << f.first << ",";
            std::cerr << "key " << f.first << "," << std::endl;
          }
          record_offsets_.emplace_back(std::vector<int32>({-1}));
          std::abort(); // hard fail
        } else {
          VLOG(4) << "Found index_source_filename for " << f;
          for (const auto& o: it->second) {
            VLOG(4) << "offset " << o;
          }
          record_offsets_.emplace_back(cumsum(it->second));
        }
      }
    } else {
      if (record_offsets_.size() < filenames_.size()) {
        VLOG(ERROR) << "Record offsets of size " << record_offsets_.size() <<
          " but there are " << filenames_.size() << " filenames.";
        std::abort();
      }
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
    return *dtypes;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({{}});
    return *shapes;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return Status::OK();
  }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* filenames = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
    Node* compression_type = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(compression_type_, &compression_type));
    Node* buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(options_.buffer_size, &buffer_size));
    Node* scan_groups = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(scan_groups_, &scan_groups));
    Node* index_source_filename = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(index_source_filename_, &index_source_filename));
    Node* metadata_output_type = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(metadata_output_type_, &metadata_output_type));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {filenames, compression_type, buffer_size, scan_groups,
        index_source_filename, metadata_output_type}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
      allocator_cache.request(pcr_state_);
    }
    ~Iterator() {
      allocator_cache.release(pcr_state_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      out_tensors->reserve(1);
      mutex_lock l(mu_);
      int32 n_scans_ = dataset()->scan_groups_;
      if (n_scans_ < 0) {
        std::cerr << "n_scans is not valid: " << n_scans_ << std::endl;
        VLOG(2) << "n_scans is not valid: " << n_scans_;
        n_scans_ = 10;
      }
      const bool add_labels_first = dataset()->metadata_output_type_ == "labels_first";
      do {
        // We are currently processing a file, so try to read the next record.
        if (reader_) {
          out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                    TensorShape({}));
          if (!current_record_index_ || !pcr_data_.serialized_data.size()) {
            // Refresh data on new record or if no data exists
            // TODO(mkuchnik): If metadata is not needed, seek to first scan
            // group
            tstring& buf_str = pcr_state_.buf_str;
            Status s =
                reader_->ReadRecord(n_scans_, &buf_str);
            TF_RETURN_IF_ERROR(s);
            if (s.ok()) {
              static monitoring::CounterCell* bytes_counter =
                  metrics::GetTFDataBytesReadCounter(kDatasetType);
              bytes_counter->IncrementBy(
                  buf_str.size());
            }
            const char* buf = buf_str.c_str();
            reader_.reset();
            VLOG(3) << "PCR file index: " << current_file_index_;
            const std::vector<int32> record_offsets =
              dataset()->record_offsets_.at(current_file_index_);
            VLOG(3) << "PCR file decoding: " << current_file_index_;
            progressive_compressed_record::generalized_parse_with_preallocated_buffs(
                  buf, record_offsets, n_scans_, pcr_state_, pcr_data_, add_labels_first);
          }
          VLOG(3) << "PCR data file:record index: " << current_record_index_
                  << ":" << current_file_index_;
          out_tensors->back().scalar<tstring>()().swap(
              pcr_data_.serialized_data.at(current_record_index_));

          *end_of_sequence = false;
          ++current_record_index_;

          const auto N_images = pcr_data_.serialized_data.size();
          if (current_record_index_ >= N_images) {
            current_record_index_ = 0;
            ResetStreamsLocked();
            ++current_file_index_;
            pcr_data_.serialized_data.clear();
          }

          return Status::OK();
        }

        // Iteration ends when there are no more files to process.
        if (current_file_index_ == dataset()->filenames_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      } while (true);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(PCR::kCurrentFileIndex),
                                             current_file_index_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(PCR::kCurrentRecordIndex), current_record_index_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      ResetStreamsLocked();
      int64 current_file_index;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(PCR::kCurrentFileIndex),
                                            &current_file_index));
      current_file_index_ = size_t(current_file_index);
      if (reader->Contains(full_name(PCR::kCurrentRecordIndex))) {
        int64 current_record_index;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name(PCR::kCurrentRecordIndex),
                               &current_record_index));

        if (current_file_index_ < dataset()->filenames_.size()) {
          // Sometimes a restored iterator is already done and will end in
          // GetNext()
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        }
        current_record_index_ = size_t(current_record_index);
      }
      return Status::OK();
    }

   private:
    // Sets up reader streams to read from the file at `current_file_index_`.
    Status SetupStreamsLocked(Env* env) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (current_file_index_ >= dataset()->filenames_.size()) {
        return errors::InvalidArgument(
            "current_file_index_:", current_file_index_,
            " >= filenames_.size():", dataset()->filenames_.size());
      }

      // Actually move on to next file.
      const string& next_filename = dataset()->filenames_.at(
          current_file_index_);
      const auto& record_offsets = dataset()->record_offsets_.at(
          current_file_index_);
      TF_RETURN_IF_ERROR(
          env->NewRandomAccessFile(next_filename, &file_));
      reader_ = absl::make_unique<io::SequentialProgressiveCompressedRecordReader>(
          file_.get(), record_offsets, dataset()->options_);
      return Status::OK();
    }

    // Resets all reader streams.
    void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      reader_.reset();
      file_.reset();
    }

    mutex mu_;
    size_t current_file_index_ TF_GUARDED_BY(mu_) = 0;
    size_t current_record_index_ TF_GUARDED_BY(mu_) = 0;
    progressive_compressed_record::Generalized_PCR_Data pcr_data_ TF_GUARDED_BY(mu_);
    progressive_compressed_record::PCR_State pcr_state_ TF_GUARDED_BY(mu_);

    // `reader_` will borrow the object that `file_` points to, so
    // we must destroy `reader_` before `file_`.
    std::unique_ptr<RandomAccessFile> file_ TF_GUARDED_BY(mu_);
    std::unique_ptr<io::SequentialProgressiveCompressedRecordReader> reader_ TF_GUARDED_BY(mu_);
  };

  const std::vector<string> filenames_;
  std::vector< std::vector<int32> > record_offsets_;
  const tstring compression_type_;
  io::ProgressiveCompressedRecordReaderOptions options_;
  const int32 scan_groups_;
  const tstring index_source_filename_;
  const tstring metadata_output_type_;
};

ProgressiveCompressedRecordDatasetOp::ProgressiveCompressedRecordDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {}

void ProgressiveCompressedRecordDatasetOp::MakeDataset(OpKernelContext* ctx,
                                    DatasetBase** output) {
  const Tensor* filenames_tensor;
  OP_REQUIRES_OK(ctx, ctx->input(kFileNames, &filenames_tensor));
  OP_REQUIRES(
      ctx, filenames_tensor->dims() <= 1,
      errors::InvalidArgument("`filenames` must be a scalar or a vector."));

  std::vector<string> filenames;
  filenames.reserve(filenames_tensor->NumElements());
  for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
    VLOG(2) << "Reading file: " << filenames_tensor->flat<tstring>()(i);
    filenames.push_back(filenames_tensor->flat<tstring>()(i));
  }

  tstring compression_type;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, kCompressionType,
                                                   &compression_type));

  int64 buffer_size = -1;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(ctx, buffer_size >= 0,
              errors::InvalidArgument(
                  "`buffer_size` must be >= 0 (0 == no buffering)"));

  int32 scan_groups = -1;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int32>(ctx, kScanGroups, &scan_groups));

  tstring index_source_filename;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<tstring>(ctx, kIndexSourceFilename, &index_source_filename));

  tstring metadata_output_type;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<tstring>(ctx, kMetadataOutputType, &metadata_output_type));
  OP_REQUIRES(ctx, validate_metadata_output_type(metadata_output_type),
              errors::InvalidArgument(
                  "`metadata_output_type` must be '' or 'labels_first'"));

  std::vector< std::vector<int> > record_offsets;

  *output =
      new Dataset(
          ctx, std::move(filenames), compression_type, buffer_size,
          scan_groups, index_source_filename, metadata_output_type,
          record_offsets);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ProgressiveCompressedRecordDataset").Device(DEVICE_CPU),
                        ProgressiveCompressedRecordDatasetOp);
}  // namespace
}  // namespace data
}
