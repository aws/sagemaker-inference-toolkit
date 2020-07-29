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
from mock import Mock, mock_open, patch
import pytest

from sagemaker_inference.utils import (
    parse_accept,
    read_file,
    retrieve_content_type_header,
    write_file,
)

TEXT = "text"
CONTENT_TYPE = "content_type"


@patch("sagemaker_inference.utils.open", new_callable=mock_open, read_data=TEXT)
def test_read_file(with_open):
    path = Mock()

    result = read_file(path)

    with_open.assert_called_once_with(path, "r")
    with_open().read.assert_called_once_with()
    assert TEXT == result


@patch("sagemaker_inference.utils.open", new_callable=mock_open, read_data=TEXT)
def test_read_file_mode(with_open):
    path = Mock()
    mode = Mock()

    result = read_file(path, mode)

    with_open.assert_called_once_with(path, mode)
    with_open().read.assert_called_once_with()
    assert result == TEXT


@patch("sagemaker_inference.utils.open", new_callable=mock_open)
def test_write_file(with_open):
    path = Mock()
    data = Mock()

    write_file(path, data)

    with_open.assert_called_once_with(path, "w")
    with_open().write.assert_called_once_with(data)


@patch("sagemaker_inference.utils.open", new_callable=mock_open)
def test_write_file_mode(with_open):
    path = Mock()
    data = Mock()
    mode = Mock()

    write_file(path, data, mode)

    with_open.assert_called_once_with(path, mode)
    with_open().write.assert_called_once_with(data)


@pytest.mark.parametrize(
    "content_type_key", ["Content-Type", "Content-type", "content-type", "ContentType"]
)
def test_content_type_header(content_type_key):
    request_property = {content_type_key: CONTENT_TYPE}

    result = retrieve_content_type_header(request_property)

    assert result == CONTENT_TYPE


@pytest.mark.parametrize(
    "input, expected",
    [
        ("application/json", ["application/json"]),
        ("application/json, text/csv", ["application/json", "text/csv"]),
        ("application/json,text/csv", ["application/json", "text/csv"]),
    ],
)
def test_parse_accept(input, expected):
    actual = parse_accept(input)
    assert actual == expected
