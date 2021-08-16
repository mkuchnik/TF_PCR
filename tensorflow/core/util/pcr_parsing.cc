#include "tensorflow/core/util/pcr_parsing.h"

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include <sstream>

// TODO disable for production
//#define PCR_DEBUG
//#define NO_PROCESS_PCR
//#define NO_UNPACK_PCR
//#define PCR_DEBUG_TIMINGS

//#define PCR_READ_FLAGS O_RDONLY | O_DIRECT
//#define PCR_READ_FLAGS O_RDONLY

#ifdef PCR_DEBUG
#include <chrono>
#endif

namespace tensorflow {
namespace progressive_compressed_record {

PCR_Data::PCR_Data() {

}
PCR_Data::PCR_Data(const std::vector< tstring >& imgs,
                   const std::vector< int32 >& lbls) : images(imgs), labels(lbls) {

}

PCR_Data parse_no_arena(
    const char* buf,
    std::vector<int32> record_offsets,
    int32 n_scans
);

// https://stackoverflow.com/questions/19343205/c-concatenating-file-and-line-macros
#define S1(x) #x
#define S2(x) S1(x)
#define LOCATION __FILE__ " : " S2(__LINE__)
#define ERROR_MSG() LOCATION

// Protobuf parse failures
void parse_failure(int err_code, const std::string& msg) {
  const std::string err_msg = "Failed to parse. Error " + \
                               std::to_string(err_code) + ". " + msg;
  std::cerr << err_msg << std::endl;
  std::abort();
}
void parse_failure(int err_code) {
  return parse_failure(err_code, "");
}


PCR_Data fake_data() {
  std::vector<int32> Y;
  std::vector< tstring > all_image_bytes;
  auto pair = PCR_Data(all_image_bytes, Y);
  return pair;
}

PCR_Data parse_no_arena(
    const char* buf,
    std::vector<int32> record_offsets,
    int32 n_scans
    ) {
  #ifdef NO_PROCESS_PCR
  {
  return fake_data();
  }
  #endif
  assert(n_scans >= 0);
  assert(static_cast<size_t>(n_scans) < record_offsets.size());

  MetadataRecord metarecord;
  std::vector<ScanGroup> scangroups(n_scans);
  std::size_t last_offset = 0;
  std::size_t expected_bytes = 0; // We use the number of bytes seen as a proxy for image size
  for (int32 i = 0; i <= n_scans; i++) {
    std::size_t offset = record_offsets.at(i);
    std::size_t size = offset - last_offset;
    std::string content = std::string(buf + last_offset, size);
    #ifdef PCR_DEBUG
    tout << "content " << last_offset << " " << offset <<  std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    #endif
    if (i == 0) {
      // First record is metadata
      int32 ret = metarecord.ParseFromString(content);
      if (!ret) {
        parse_failure(ret, ERROR_MSG());
      }
    } else {
      expected_bytes += offset;
      ScanGroup& scangroup = scangroups.at(i - 1);
      int32 ret = scangroup.ParseFromString(content);
      if (!ret) {
        parse_failure(ret, ERROR_MSG());
      }
    }
    #ifdef PCR_DEBUG
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    tout << "duration decode " << i << " = " << duration.count() << std::endl;
    #endif
    last_offset = offset;
  }

  #ifdef NO_UNPACK_PCR
  {
  return fake_data();
  }
  #endif

  // Normalize
  expected_bytes = expected_bytes / n_scans + 1;

  const size_t N_images = metarecord.labels().size();

  std::vector<int32> Y;
  Y.reserve(N_images);
  for (auto& label: metarecord.labels()) {
    Y.push_back(label);
  }

  // Preallocate
  std::vector< tstring > all_image_bytes(N_images);

  // TODO(mkuchnik): StringPiece may be safer for partial string references
  for (const auto& scangroup: scangroups) {
    // This contains N==len(Y) partial images
    const auto& grouped_partial = scangroup.image_bytes();
    assert(grouped_partial.size() >= 0);
    assert(static_cast<size_t>(grouped_partial.size()) == N_images);
    for (int32 i = 0; i < grouped_partial.size(); i++) {
      tstring partial_image;
      partial_image.assign_as_view(
          grouped_partial[i].c_str(),
          grouped_partial[i].size());
      std::size_t n_new_bytes = partial_image.size();
      all_image_bytes[i].reserve(all_image_bytes[i].size() + n_new_bytes);
      all_image_bytes[i].append(partial_image);
    }
  }

  const unsigned char end_of_image[2] = {0xFF, 0xD9};
  for (auto& image_bytes: all_image_bytes) {
    std::size_t str_len = image_bytes.length();
    if ((image_bytes[str_len-2] != end_of_image[0]) ||
        (image_bytes[str_len-1] != end_of_image[1])) {
       image_bytes.append((const char*) end_of_image, 2);
    }
  }

  auto pair = PCR_Data(all_image_bytes, Y);
  return pair;
}

std::map< std::string, std::vector<int32> > parse_PCR_index(const std::string& content) {
  DatasetOffsetsIndex index;
  std::map< std::string, std::vector<int32> > index_map;
  int32 ret = index.ParseFromString(content);
  if (!ret) {
    parse_failure(ret, ERROR_MSG());
  }
  for (size_t i = 0; i < index.records_size(); i++) {
    const RecordOffsetsIndex& rec_index = index.records(i);
    const std::string& name = rec_index.name();
    std::vector<int32> offsets;
    offsets.reserve(rec_index.offsets_size());
    for (auto& off : rec_index.offsets()) {
      offsets.push_back(off);
    }
    index_map.insert( std::pair<std::string, std::vector<int32> >(name, offsets) );
  }
  return index_map;
}

// Gets labels into vector
std::vector<int32> extract_labels(const MetadataRecord& metarecord) {
  const size_t N_images = metarecord.labels().size();
  std::vector<int32> Y(N_images);
  Y.reserve(N_images);
  assert (metarecord.labels().size() >= 0);
  for (std::size_t i = 0; i < static_cast<size_t>(metarecord.labels().size()); i++) {
    const auto& label = metarecord.labels()[i];
    Y.at(i) = label;
  }
  return Y;
}

void extract_images(std::vector< tstring >& all_image_bytes, std::vector<ScanGroup>& scangroups, size_t N_images) {
  for (const auto& scangroup: scangroups) {
    // This contains N==len(Y) partial images
    const auto& grouped_partial = scangroup.image_bytes();
    assert(grouped_partial.size() >= 0);
    assert(static_cast<size_t>(grouped_partial.size()) == N_images);
    for (std::size_t i = 0; i < static_cast<size_t>(grouped_partial.size()); i++) {
      all_image_bytes[i].append(grouped_partial[i]);
    }
  }
}

// For preallocation of containers
std::vector< std::size_t > calculate_sizes_of_images(
    std::vector<ScanGroup>& scangroups, std::size_t N_images) {
  std::vector< std::size_t > all_image_bytes_sizes(N_images, 0);
  for (auto scangroup: scangroups) {
    // This contains N==len(Y) partial images
    const auto& grouped_partial = scangroup.image_bytes();
    for (std::size_t i = 0; i < static_cast<size_t>(grouped_partial.size()); i++) {
      all_image_bytes_sizes[i] += grouped_partial[i].size();
    }
  }
  return all_image_bytes_sizes;
}

// Sizes inner tensor
void preallocate_vector(std::vector< tstring >& vec, const std::vector< std::size_t >& inner_sizes, const size_t extra_terminal_bytes) {
  const auto N_images = vec.size();
  assert(N_images == inner_sizes.size());
  for (std::size_t i = 0; i < N_images; i++) {
    vec[i].reserve(inner_sizes[i] + extra_terminal_bytes);
  }
}

void add_terminal_chars(std::vector< tstring >& all_image_bytes) {
  const unsigned char end_of_image[2] = {0xFF, 0xD9};
  for (std::size_t i = 0; i < all_image_bytes.size(); i++) {
    auto& image_bytes = all_image_bytes[i];
    image_bytes.append((const char*) end_of_image, 2);
  }
}

PCR_Data parse_with_preallocated_buffs(
    const char* buf,
    const std::vector<int32>& record_offsets,
    int32 n_scans,
    PCR_State& state
    ) {
  // TODO arena is not used
  // google::protobuf::Arena* arena=NULL
  #ifdef NO_PROCESS_PCR
  {
  return fake_data();
  }
  #endif
  MetadataRecord& metarecord = state.metarecord;
  std::vector<ScanGroup>& scangroups = state.scangroups;

  assert (buf != NULL);

  assert(n_scans >= 0);
  if (scangroups.size() != static_cast<size_t>(n_scans)) {
    scangroups.resize(n_scans);
  }

  std::size_t last_offset = 0;
  std::size_t expected_bytes = 0; // We use the number of bytes seen as a proxy for image size
  for (int32 i = 0; i <= n_scans; i++) {
    std::size_t offset = record_offsets.at(i);
    std::size_t size = offset - last_offset;
    std::string content = std::string(buf + last_offset, size);
    #ifdef PCR_DEBUG
    tout << "content " << last_offset << " " << offset <<  std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    #endif
    if (i == 0) {
      // First record is metadata
      auto ret = metarecord.ParseFromString(content);
      if (!ret) {
        parse_failure(ret, ERROR_MSG());
      }
    } else {
      expected_bytes += offset;
      ScanGroup& scangroup = scangroups.at(i-1);
      auto ret = scangroup.ParseFromString(content);
      if (!ret) {
        parse_failure(ret, ERROR_MSG());
      }
    }
    #ifdef PCR_DEBUG
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    tout << "duration decode " << i << " = " << duration.count() << std::endl;
    #endif
    last_offset = offset;
  }

  #ifdef NO_UNPACK_PCR
  {
  return fake_data();
  }
  #endif

  // Normalize
  expected_bytes = expected_bytes / n_scans + 1;
  const size_t N_images = metarecord.labels().size();

  // Get labels
  std::vector<int32> Y = extract_labels(metarecord);

  // Preallocate
  std::vector< tstring > all_image_bytes(N_images);
  std::vector< std::size_t > all_image_bytes_sizes = calculate_sizes_of_images(
      scangroups, N_images);

  // Preallocate
  // NOTE(mkuchnik): We only need 2 bytes for end and 4 bytes for label
  const size_t extra_terminal_bytes = 6;
  preallocate_vector(all_image_bytes, all_image_bytes_sizes, extra_terminal_bytes);

  // Copy data
  extract_images(all_image_bytes, scangroups, N_images);

  // Add terminal chars
  // NOTE(mkuchnik): We save space for labels
  add_terminal_chars(all_image_bytes);

  // TODO(mkuchnik): This is a copy
  const auto pair = PCR_Data(all_image_bytes, Y);
  return pair;
}

void pack_labels_into_tensor(tstring& data_tensor, int32 label) {
  // TODO(mkuchnik): Hacky byte magic
  tstring label_payload(reinterpret_cast<char*>(&label), 4);
  data_tensor.append(label_payload);
}

void generalized_parse_with_preallocated_buffs(
    const char* buf,
    const std::vector<int32>& record_offsets,
    int32 n_scans,
    PCR_State& state,
    Generalized_PCR_Data& pcr_data,
    bool pack_metadata_first
    ) {
  // TODO arena is not used
  // google::protobuf::Arena* arena=NULL
  #ifdef NO_PROCESS_PCR
  {
  return fake_data();
  }
  #endif
  MetadataRecord& metarecord = state.metarecord;
  std::vector<ScanGroup>& scangroups = state.scangroups;

  assert (buf != NULL);

  assert(n_scans >= 0);
  if (scangroups.size() != static_cast<size_t>(n_scans)) {
    scangroups.resize(n_scans);
  }

  std::size_t last_offset = 0;
  std::size_t expected_bytes = 0; // We use the number of bytes seen as a proxy for image size
  for (int32 i = 0; i <= n_scans; i++) {
    std::size_t offset = record_offsets.at(i);
    std::size_t size = offset - last_offset;
    std::string content = std::string(buf + last_offset, size);
    #ifdef PCR_DEBUG
    tout << "content " << last_offset << " " << offset <<  std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    #endif
    if (i == 0) {
      // First record is metadata
      auto ret = metarecord.ParseFromString(content);
      if (!ret) {
        parse_failure(ret, ERROR_MSG());
      }
    } else {
      expected_bytes += offset;
      ScanGroup& scangroup = scangroups.at(i-1);
      auto ret = scangroup.ParseFromString(content);
      if (!ret) {
        parse_failure(ret, ERROR_MSG());
      }
    }
    #ifdef PCR_DEBUG
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    tout << "duration decode " << i << " = " << duration.count() << std::endl;
    #endif
    last_offset = offset;
  }

  #ifdef NO_UNPACK_PCR
  {
  return fake_data();
  }
  #endif

  // Normalize
  expected_bytes = expected_bytes / n_scans + 1;

  const size_t N_images = metarecord.labels().size();

  // Get labels
  std::vector<int32> Y = extract_labels(metarecord);

  // Preallocate
  auto& all_image_bytes = pcr_data.serialized_data;
  all_image_bytes.resize(N_images);
  std::vector< std::size_t > all_image_bytes_sizes = calculate_sizes_of_images(
      scangroups, N_images);

  // Preallocate
  // NOTE(mkuchnik): We only need 2 bytes for end and 4 bytes for label
  if (pack_metadata_first) {
    preallocate_vector(all_image_bytes, all_image_bytes_sizes, 6);
  } else {
    preallocate_vector(all_image_bytes, all_image_bytes_sizes, 2);
  }

  // Pack metadata first
  if (pack_metadata_first) {
    for (size_t i = 0; i < N_images; i++) {
      pack_labels_into_tensor(all_image_bytes[i], Y[i]);
    }
  }

  // Copy data
  extract_images(all_image_bytes, scangroups, N_images);

  // Add terminal chars
  // NOTE(mkuchnik): We save space for labels
  add_terminal_chars(all_image_bytes);
}

}  // namespace progressive_compressed_record
}  // namespace tensorflow
