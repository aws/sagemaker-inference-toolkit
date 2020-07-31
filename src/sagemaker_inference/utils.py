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
"""This module contains utility functions related to reading files,
writing files, and retrieving information from requests.
"""
from __future__ import absolute_import

import re

CONTENT_TYPE_REGEX = re.compile("^[Cc]ontent-?[Tt]ype")


def read_file(path, mode="r"):
    """Read data from a file.

    Args:
        path (str): path to the file.
        mode (str): mode which the file will be open.

    Returns:
        (str): contents of the file.

    """
    with open(path, mode) as f:
        return f.read()


def write_file(path, data, mode="w"):  # type: (str, str, str) -> None
    """Write data to a file.

    Args:
        path (str): path to the file.
        data (str): data to be written to the file.
        mode (str): mode which the file will be open.

    """
    with open(path, mode) as f:
        f.write(data)


def retrieve_content_type_header(request_property):
    """Retrieve Content-Type header from incoming request.

    This function handles multiple spellings of Content-Type based on the presence of
    the dash and initial capitalization in each respective word.

    Args:
        request_property (dict): incoming request metadata

    Returns:
        (str): the request content type.

    """
    for key in request_property:
        if CONTENT_TYPE_REGEX.match(key):
            return request_property[key]

    return None


def parse_accept(accept):
    """Parses the Accept header sent with a request.

    Args:
        accept (str): the value of an Accept header.

    Returns:
        (list): A list containing the MIME types that the client is able to
            understand.
    """
    return accept.replace(" ", "").split(",")


def remove_crlf(illegal_string):
    """Removes characters prohibited by the MMS dependency Netty.

    https://github.com/netty/netty/issues/8312

    Args:
        illegal_string: The string containing prohibited characters.

    Returns:
        str: The input string with the prohibited characters removed.
    """
    prohibited = ("\r", "\n")

    sanitized_string = illegal_string

    for character in prohibited:
        sanitized_string = sanitized_string.replace(character, " ")

    return sanitized_string
