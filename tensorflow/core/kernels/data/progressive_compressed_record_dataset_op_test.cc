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

#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/example/progressive_compressed_record.pb.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "progressive_compressed_record_dataset";

class ProgressiveCompressedRecordDatasetParams : public DatasetParams {
 public:
  ProgressiveCompressedRecordDatasetParams(
      std::vector<tstring> filenames,
      tstring index_filename,
      int64 scan_group,
      string node_name)
      : DatasetParams({DT_STRING}, {PartialTensorShape({})},
                      std::move(node_name)),
        filenames_(std::move(filenames)),
        compression_type_(CompressionType::UNCOMPRESSED),
        buffer_size_(10),
        scan_group_(scan_group),
        index_filename_(index_filename),
        metadata_output_type_("") {}

  std::vector<Tensor> GetInputTensors() const override {
    int num_files = filenames_.size();
    return {
        CreateTensor<tstring>(TensorShape({num_files}), filenames_),
        CreateTensor<tstring>(TensorShape({}), {ToString(compression_type_)}),
        CreateTensor<int64>(TensorShape({}), {buffer_size_}),
        CreateTensor<int64>(TensorShape({}), {scan_group_}),
        CreateTensor<tstring>(TensorShape({}), {index_filename_}),
        CreateTensor<tstring>(TensorShape({}), {metadata_output_type_})
    };
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    *input_names = {
        // NOTE(mkuchnik): We use default params for other args
        ProgressiveCompressedRecordDatasetOp::kFileNames,
        ProgressiveCompressedRecordDatasetOp::kCompressionType,  // N/A
        ProgressiveCompressedRecordDatasetOp::kBufferSize,  // N/A
        ProgressiveCompressedRecordDatasetOp::kScanGroups,
        ProgressiveCompressedRecordDatasetOp::kIndexSourceFilename,
        ProgressiveCompressedRecordDatasetOp::kMetadataOutputType
    };
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {};
    return Status::OK();
  }

  string dataset_type() const override {
    return ProgressiveCompressedRecordDatasetOp::kDatasetType;
  }

 private:
  std::vector<tstring> filenames_;
  CompressionType compression_type_;
  int64 buffer_size_;
  int64 scan_group_;
  tstring index_filename_;
  tstring metadata_output_type_;
};

class ProgressiveCompressedRecordDatasetOpTest : public DatasetOpsTestBase {};

Status CreateTestFiles(const std::vector<tstring>& filenames,
                       const tstring& index_filename,
                       const std::vector< std::vector<int32> >& labels,
                       const std::vector< std::vector<std::vector<tstring>> >& records,
                       CompressionType compression_type,
                       bool relative_indexing) {
  // TODO(mkuchnik): Use one index file for all filenames
  if (filenames.size() != records.size() || filenames.size() != labels.size()) {
    return tensorflow::errors::InvalidArgument(
        "The number of files does not match with the contents");
  }
  DatasetOffsetsIndex full_index;
  std::vector<RecordOffsetsIndex> indices;
  for (int i = 0; i < filenames.size(); ++i) {
    CompressionParams params;
    // TODO(mkuchnik): Overwrite these params for PCRWriter
    params.output_buffer_size = 10;
    // TODO(mkuchnik): Probably don't want compression
    params.compression_type = compression_type;
    std::vector<size_t> index_offsets;
    std::vector<Example> examples(labels.at(i).size());
    std::vector< std::vector<absl::string_view> > sv_records;
    for (const auto& r : records.at(i)) {
      std::vector<absl::string_view> rr(r.begin(), r.end());
      sv_records.push_back(rr);
    }
    auto filename = std::string{ filenames.at(i) };
    TF_RETURN_IF_ERROR(WriteDataToProgressiveCompressedRecordFile(
          /*filename=*/filename,
          /*index_offsets=*/index_offsets,
          /*labels=*/labels.at(i),
          /*records=*/sv_records,
          /*examples=*/examples,
          /*params=*/params)
    );
    RecordOffsetsIndex index_record;
    if (relative_indexing) {
      // Use relative path for filename
      const auto slash_it = filename.find_last_of("/\\");
      if (slash_it != string::npos) {
        const auto base_filename = filename.substr(slash_it + 1); // strip path
        index_record.set_name(base_filename);
      }
    } else{
      index_record.set_name(filename);
    }
    *index_record.mutable_offsets() = {index_offsets.begin(),
                                       index_offsets.end()};
    indices.push_back(index_record);
  }
  *full_index.mutable_records() = {indices.begin(), indices.end()};
  std::string payload;
  if (!full_index.SerializeToString(&payload)) {
    LOG(ERROR) << "Could not serialize full_index";
    return Status(::tensorflow::error::INTERNAL,
                  "Could not serialize full_index.");
  }
  TF_RETURN_IF_ERROR(WriteDataToFile(index_filename, payload.data()));
  return Status::OK();
}

// Test case 1: single text files with 3 scans.
ProgressiveCompressedRecordDatasetParams ProgressiveCompressedRecordDatasetParams1() {
  const std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/pcr_record_UNCOMPRESSED1_1")};
  const tstring index_filename = absl::StrCat(testing::TmpDir(),
                                        "/pcr_record_index_UNCOMPRESSED1_1");
  // Note(mkuchnik): We have one content/label per filename
  // The 3 levels of nesting are: file, record, and scan
  const std::vector< std::vector<std::vector<tstring> > > contents = { { {"111", "222", "333"} } };
  const std::vector<std::vector<int32>> labels = { {1} };
  int64 scan_group = 3;
  bool relative_indexing = false;
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  Status s = CreateTestFiles(
      filenames, index_filename, labels, contents, compression_type,
      relative_indexing);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ")
                  << ": " << s;
  }
  return ProgressiveCompressedRecordDatasetParams(
      filenames,
      /*index_filename=*/index_filename,
      /*scan_groups=*/scan_group,
      /*node_name=*/kNodeName);
}

// Test case 2: single text files with 3 scans.
ProgressiveCompressedRecordDatasetParams ProgressiveCompressedRecordDatasetParams2() {
  const std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/pcr_record_UNCOMPRESSED2_1")};
  const tstring index_filename = absl::StrCat(testing::TmpDir(),
                                        "/pcr_record_index_UNCOMPRESSED2_1");
  // Note(mkuchnik): We have one content/label per filename
  // The 3 levels of nesting are: file, record, and scan
  const std::vector< std::vector<std::vector<tstring> > > contents = {
    { {"a", "bb", "ccc"} } };
  const std::vector<std::vector<int32>> labels = { {0} };
  int64 scan_group = 1;
  bool relative_indexing = false;
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  Status s = CreateTestFiles(
      filenames, index_filename, labels, contents, compression_type,
      relative_indexing);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ")
                  << ": " << s;
  }
  return ProgressiveCompressedRecordDatasetParams(
      filenames,
      /*index_filename=*/index_filename,
      /*scan_groups=*/scan_group,
      /*node_name=*/kNodeName);
}

// Test case 3: single text files with 3 scans with relative index.
ProgressiveCompressedRecordDatasetParams ProgressiveCompressedRecordDatasetParams3() {
  const std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/pcr_record_UNCOMPRESSED3_1")};
  const tstring index_filename = absl::StrCat(testing::TmpDir(),
                                        "/pcr_record_index_UNCOMPRESSED3_1");
  // Note(mkuchnik): We have one content/label per filename
  // The 3 levels of nesting are: file, record, and scan
  const std::vector< std::vector<std::vector<tstring> > > contents = {
    { {"111", "222", "333"} } };
  const std::vector<std::vector<int32>> labels = { {1} };
  int64 scan_group = 3;
  bool relative_indexing = true;
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  Status s = CreateTestFiles(
      filenames, index_filename, labels, contents, compression_type,
      relative_indexing);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ")
                  << ": " << s;
  }
  return ProgressiveCompressedRecordDatasetParams(
      filenames,
      /*index_filename=*/index_filename,
      /*scan_groups=*/scan_group,
      /*node_name=*/kNodeName);
}


// Test case 4: single text files with 3 scans with multiple records.
ProgressiveCompressedRecordDatasetParams ProgressiveCompressedRecordDatasetParams4() {
  const std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/pcr_record_UNCOMPRESSED4_1")};
  const tstring index_filename = absl::StrCat(testing::TmpDir(),
                                        "/pcr_record_index_UNCOMPRESSED4_1");
  // Note(mkuchnik): We have one content/label per filename
  // The 3 levels of nesting are: file, record, and scan
  const std::vector< std::vector<std::vector<tstring> > > contents = {
    { {"111", "222", "333"}, {"a", "bb", "ccc"} } };
  const std::vector<std::vector<int32>> labels = { {1, 2} };
  int64 scan_group = 3;
  bool relative_indexing = false;
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  Status s = CreateTestFiles(
      filenames, index_filename, labels, contents, compression_type,
      relative_indexing);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ")
                  << ": " << s;
  }
  return ProgressiveCompressedRecordDatasetParams(
      filenames,
      /*index_filename=*/index_filename,
      /*scan_groups=*/scan_group,
      /*node_name=*/kNodeName);
}

// Test case 5: multiple text files with 5 scans with multiple records.
ProgressiveCompressedRecordDatasetParams ProgressiveCompressedRecordDatasetParams5() {
  const std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/pcr_record_UNCOMPRESSED5_1"),
      absl::StrCat(testing::TmpDir(), "/pcr_record_UNCOMPRESSED5_2"),
      absl::StrCat(testing::TmpDir(), "/pcr_record_UNCOMPRESSED5_3")
  };
  const tstring index_filename = absl::StrCat(testing::TmpDir(),
                                        "/pcr_record_index_UNCOMPRESSED5_1");
  // Note(mkuchnik): We have one content/label per filename
  // The 3 levels of nesting are: file, record, and scan
  const std::vector< std::vector<std::vector<tstring> > > contents = {
    { {"111", "222", "333"} },
    { {"a", "bb", "ccc"} },
    { {"alone", "", ""} } };
  const std::vector<std::vector<int32>> labels = { {1}, {2}, {-1} };
  int64 scan_group = 3;
  bool relative_indexing = false;
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  Status s = CreateTestFiles(
      filenames, index_filename, labels, contents, compression_type,
      relative_indexing);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ")
                  << ": " << s;
  }
  return ProgressiveCompressedRecordDatasetParams(
      filenames,
      /*index_filename=*/index_filename,
      /*scan_groups=*/scan_group,
      /*node_name=*/kNodeName);
}


// Test case 6: many outputs.
ProgressiveCompressedRecordDatasetParams ProgressiveCompressedRecordDatasetParams6() {
  const std::vector<tstring> filenames = {
      absl::StrCat(testing::TmpDir(), "/pcr_record_UNCOMPRESSED6_1"),
      absl::StrCat(testing::TmpDir(), "/pcr_record_UNCOMPRESSED6_2")
  };
  const tstring index_filename = absl::StrCat(testing::TmpDir(),
                                        "/pcr_record_index_UNCOMPRESSED6_1");
  // Note(mkuchnik): We have one content/label per filename
  // The 3 levels of nesting are: file, record, and scan
  const std::vector< std::vector<std::vector<tstring> > > contents = {
    { {"1", "", ""}, {"2", "2", ""}, {"3", "3", "3"} },
    { {"a", "", ""}, {"bb", "", ""}, {"c", "c", "c"} }
  };
  const std::vector<std::vector<int32>> labels = {
    {1, 2, 3},
    {4, 5, 6}
  };
  int64 scan_group = 3;
  bool relative_indexing = false;
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  Status s = CreateTestFiles(
      filenames, index_filename, labels, contents, compression_type,
      relative_indexing);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ")
                  << ": " << s;
  }
  return ProgressiveCompressedRecordDatasetParams(
      filenames,
      /*index_filename=*/index_filename,
      /*scan_groups=*/scan_group,
      /*node_name=*/kNodeName);
}


std::string string_to_pcr(std::string s) {
  // Adds extra characters at end
  s.append({0xFF, 0xD9});
  return s;
}

// Note(mkuchnik): We expect scan 1 to be only the first element
// scan 2 will be the first 2
// scan 3 will be all 3 elements
std::vector<GetNextTestCase<ProgressiveCompressedRecordDatasetParams>>
    GetNextTestCases() {
  return {
      {/*dataset_params=*/ProgressiveCompressedRecordDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), { {string_to_pcr("111222333")} }
           )},
      {/*dataset_params=*/ProgressiveCompressedRecordDatasetParams2(),
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), { {string_to_pcr("a")} }
           )},
      {/*dataset_params=*/ProgressiveCompressedRecordDatasetParams3(),
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), { {string_to_pcr("111222333")} }
           )},
      {/*dataset_params=*/ProgressiveCompressedRecordDatasetParams4(),
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), { {string_to_pcr("111222333")},
                              {string_to_pcr("abbccc")} }
           )},
      {/*dataset_params=*/ProgressiveCompressedRecordDatasetParams5(),
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), { {string_to_pcr("111222333")},
                              {string_to_pcr("abbccc")},
                              {string_to_pcr("alone")} }
           )},
      {/*dataset_params=*/ProgressiveCompressedRecordDatasetParams6(),
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), {{string_to_pcr("1")},
                             {string_to_pcr("22")},
                             {string_to_pcr("333")},
                             {string_to_pcr("a")},
                             {string_to_pcr("bb")},
                             {string_to_pcr("ccc")} }
           )}
  };
}

ITERATOR_GET_NEXT_TEST_P(ProgressiveCompressedRecordDatasetOpTest,
                         ProgressiveCompressedRecordDatasetParams,
                         GetNextTestCases())

TEST_F(ProgressiveCompressedRecordDatasetOpTest, DatasetNodeName) {
  auto dataset_params = ProgressiveCompressedRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ProgressiveCompressedRecordDatasetOpTest, DatasetTypeString) {
  auto dataset_params = ProgressiveCompressedRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(ProgressiveCompressedRecordDatasetOp::kDatasetType)));
}

TEST_F(ProgressiveCompressedRecordDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = ProgressiveCompressedRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_STRING}));
}

TEST_F(ProgressiveCompressedRecordDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = ProgressiveCompressedRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(ProgressiveCompressedRecordDatasetOpTest, Cardinality) {
  auto dataset_params = ProgressiveCompressedRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(kUnknownCardinality));
}

TEST_F(ProgressiveCompressedRecordDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = ProgressiveCompressedRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_STRING}));
}

TEST_F(ProgressiveCompressedRecordDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = ProgressiveCompressedRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(ProgressiveCompressedRecordDatasetOpTest, IteratorPrefix) {
  auto dataset_params = ProgressiveCompressedRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ProgressiveCompressedRecordDatasetOp::kDatasetType,
      dataset_params.iterator_prefix())));
}

std::vector<
  IteratorSaveAndRestoreTestCase<ProgressiveCompressedRecordDatasetParams> >
    IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/ProgressiveCompressedRecordDatasetParams1(),
       /*breakpoints=*/{0, 1},
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), { {string_to_pcr("111222333")} })
      },
      {/*dataset_params=*/ProgressiveCompressedRecordDatasetParams6(),
       /*breakpoints=*/{0, 2, 7},
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({}), {{string_to_pcr("1")},
                             {string_to_pcr("22")},
                             {string_to_pcr("333")},
                             {string_to_pcr("a")},
                             {string_to_pcr("bb")},
                             {string_to_pcr("ccc")}})
      }
  };
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ProgressiveCompressedRecordDatasetOpTest,
                                 ProgressiveCompressedRecordDatasetParams,
                                 IteratorSaveAndRestoreTestCases())


}  // namespace
}  // namespace data
}
