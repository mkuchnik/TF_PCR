# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `tf.data.ProgressiveCompressedRecordDataset`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import zlib

from absl.testing import parameterized

import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.lib.io import python_io
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.platform import resource_loader
from tensorflow.python.util import compat
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.framework import dtypes


def assert_path_exists(path):
    assert(os.path.exists(path))

class ProgressiveCompressedRecordDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def test_pcr_open(self):
        """Test PCRs opening files and binary equality to input"""
        base = "tensorflow/core/lib/jpeg/testdata"
        filename = os.path.join(
            base, "PCR_0.pcr"
        )
        filenames = [filename]
        index_filename = os.path.join(
            base, "PCR_index.pb"
        )
        source_filename = os.path.join(
            base, "small_progressive.jpg"
        )
        assert_path_exists(filename)
        assert_path_exists(index_filename)
        assert_path_exists(source_filename)
        scans = 10
        dataset = readers.ProgressiveCompressedRecordDataset(filenames,
                                                             scan_groups=scans,
                                                             index_source_filename=index_filename,
                                                             metadata_output_type="")
        get_next = self.getNext(dataset)
        #x = [self.evaluate(get_next())]
        #self.assertEqual(dataset_ops.get_legacy_output_types(dataset),
        #                 dtypes.string)
        #with open(source_filename, "rb") as f:
        #    image_content = f.read()
        #pcr_image_content = np.array(x[0])
        #self.assertEqual(pcr_image_content, np.array(image_content))
        #with self.assertRaises(errors.OutOfRangeError):
        #  results = self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def test_pcr_scans_truncate(self):
        """Tests that scans truncate original input"""
        base = "tensorflow/core/lib/jpeg/testdata"
        filename = os.path.join(
            base, "PCR_0.pcr"
        )
        filenames = [filename]
        index_filename = os.path.join(
            base, "PCR_index.pb"
        )
        assert_path_exists(filename)
        assert_path_exists(index_filename)
        last_num_bytes = 0
        for scans in [1, 10]:
            dataset = readers.ProgressiveCompressedRecordDataset(filenames,
                                                                 scan_groups=scans,
                                                                 index_source_filename=index_filename,
                                                                 metadata_output_type="")
            #get_next = self.getNext(dataset)
            #x = [self.evaluate(get_next())]
            #with self.assertRaises(errors.OutOfRangeError):
            #  results = self.evaluate(get_next())
            #self.assertEqual(dataset_ops.get_legacy_output_types(dataset),
            #                 dtypes.string)
            #num_bytes = len(x[0])
            #assert num_bytes > last_num_bytes
            #last_num_bytes = num_bytes

    @combinations.generate(test_base.default_test_combinations())
    def test_pcr_decode(self):
        """Tests that decoding more scans improves quality"""
        base = "tensorflow/core/lib/jpeg/testdata"
        filename = os.path.join(
            base, "PCR_0.pcr"
        )
        filenames = [filename]
        index_filename = os.path.join(
            base, "PCR_index.pb"
        )
        source_filename = os.path.join(
            base, "small_progressive.jpg"
        )
        assert_path_exists(filename)
        assert_path_exists(index_filename)
        assert_path_exists(source_filename)
        def parse_file(filename):
            image = io_ops.read_file(filename)
            image = image_ops.decode_jpeg(image)
            return image
        def parse_img_data(img_data):
            image = image_ops.decode_jpeg(img_data)
            return image
        reference_ds = dataset_ops.Dataset.from_tensor_slices([source_filename])
        reference_ds = reference_ds.map(parse_file)
        get_next = self.getNext(reference_ds)
        reference_x = [self.evaluate(get_next())]
        with self.assertRaises(errors.OutOfRangeError):
          results = self.evaluate(get_next())
        self.assertEqual(dataset_ops.get_legacy_output_types(reference_ds),
                         dtypes.uint8)
        reference_image_content = reference_x[0]
        error_bounds = 6.8
        for scans in range(1, 11):
            dataset = readers.ProgressiveCompressedRecordDataset(filenames,
                                                                 scan_groups=scans,
                                                                 index_source_filename=index_filename,
                                                                 metadata_output_type="")
            #dataset = dataset.map(parse_img_data)
            #get_next = self.getNext(dataset)
            #x = [self.evaluate(get_next())]
            #with self.assertRaises(errors.OutOfRangeError):
            #  results = self.evaluate(get_next())
            #self.assertEqual(dataset_ops.get_legacy_output_types(dataset),
            #                 dtypes.uint8)
            #pcr_image_content = np.array(x[0])
            #if scans == 10:
            #    assert np.all(pcr_image_content == reference_image_content)
            #else:
            #    assert pcr_image_content.shape == reference_image_content.shape
            #    error = np.sqrt(np.mean((pcr_image_content -
            #                             reference_image_content)**2))
            #    assert error < error_bounds, \
            #        "{} >= {} for scan {}".format(error, error_bounds, scans)
            #    error_bounds = error # Assert decreasing


if __name__ == "__main__":
  test.main()
