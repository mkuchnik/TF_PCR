#ifndef PCR_PARSING_H_
#define PCR_PARSING_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/example/progressive_compressed_record.pb.h"

namespace tensorflow {
namespace progressive_compressed_record {

class PCR_Data {
  public:
    std::vector< tstring > images;
    std::vector< int32 > labels;
    PCR_Data();
    PCR_Data(const std::vector< tstring >& imgs, const std::vector< int32 >& lbls);
};

// This version repacks everything into a Protobuf
struct Generalized_PCR_Data {
  public:
    std::vector< tstring > serialized_data;
};

// Intermediate buffers to be re-used
struct PCR_State {
  public:
    MetadataRecord metarecord;
    std::vector<ScanGroup> scangroups;
    tstring buf_str; // For data loading and scratch-space
};

/**
 * Inverse function of create_progressive_compressed_tf_record
 * @param buf The data buffer
 * @param record_offsets A list of data corresponding to
 * [metadata, scan_0, scan_1,...] for
 * potentially many images
 * @param n_scans Only n_scans are required. This should match records
 * @return The bytes of JPEG images and the labels as tuple (X, Y)
 */
PCR_Data parse_no_arena(
    const char* buf,
    std::vector<int32> record_offsets,
    int32 n_scans
    );

// Optimized version of above, passing buffers around
PCR_Data parse_with_preallocated_buffs(
    const char* buf,
    const std::vector<int32>& record_offsets,
    int32 n_scans,
    PCR_State& state
    );


// Even more optimized version of above, passing buffers around and allocating
// return data in place
void generalized_parse_with_preallocated_buffs(
    const char* buf,
    const std::vector<int32>& record_offsets,
    int32 n_scans,
    PCR_State& state,
    Generalized_PCR_Data& pcr_data,
    bool pack_metadata_first
    );

std::map< std::string, std::vector<int32> > parse_PCR_index(const std::string& content);


}  // namespace progressive_compressed_record
}  // namespace tensorflow

#endif // PCR_PARSING_H_
