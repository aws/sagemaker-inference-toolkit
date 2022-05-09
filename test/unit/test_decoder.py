# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from mock import Mock, patch
import numpy as np
import pytest
import scipy.sparse
from six import BytesIO

from sagemaker_inference import content_types, decoder, errors


@pytest.mark.parametrize(
    "target",
    ([42, 6, 9], [42.0, 6.0, 9.0], ["42", "6", "9"], ["42", "6", "9"], {42: {"6": 9.0}}),
)
def test_npy_to_numpy(target):
    buffer = BytesIO()
    np.save(buffer, target)
    input_data = buffer.getvalue()

    actual = decoder._npy_to_numpy(input_data)

    np.testing.assert_equal(actual, np.array(target))


@pytest.mark.parametrize(
    "target, expected",
    [
        ("[42, 6, 9]", np.array([42, 6, 9])),
        ("[42.0, 6.0, 9.0]", np.array([42.0, 6.0, 9.0])),
        ('["42", "6", "9"]', np.array(["42", "6", "9"])),
        ('["42", "6", "9"]', np.array(["42", "6", "9"])),
    ],
)
def test_json_to_numpy(target, expected):
    actual = decoder._json_to_numpy(target)
    np.testing.assert_equal(actual, expected)

    np.testing.assert_equal(decoder._json_to_numpy(target, dtype=int), expected.astype(int))

    np.testing.assert_equal(decoder._json_to_numpy(target, dtype=float), expected.astype(float))


@pytest.mark.parametrize(
    "target, expected",
    [
        ("42\n6\n9\n", np.array([42, 6, 9])),
        ("42.0\n6.0\n9.0\n", np.array([42.0, 6.0, 9.0])),
        ("42\n6\n9\n", np.array([42, 6, 9])),
    ],
)
def test_csv_to_numpy(target, expected):
    actual = decoder._csv_to_numpy(target)
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "target",
    [
        scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]])),
        scipy.sparse.csr_matrix(np.array([[1, 0], [0, 7]])),
        scipy.sparse.coo_matrix(np.array([[6, 2], [5, 9]])),
    ],
)
def test_npz_to_sparse(target):
    buffer = BytesIO()
    scipy.sparse.save_npz(buffer, target)
    data = buffer.getvalue()
    matrix = decoder._npz_to_sparse(data)

    actual = matrix.toarray()
    expected = target.toarray()

    np.testing.assert_equal(actual, expected)


def test_decode_error():
    with pytest.raises(errors.UnsupportedFormatError):
        decoder.decode(42, content_types.OCTET_STREAM)


@pytest.mark.parametrize("content_type", [content_types.JSON, content_types.CSV, content_types.NPY])
def test_decode(content_type):
    mock_decoder = Mock()
    with patch.dict(decoder._decoder_map, {content_type: mock_decoder}, clear=True):
        decoder.decode(42, content_type)

        mock_decoder.assert_called_once_with(42)
