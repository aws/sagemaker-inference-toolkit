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
"""This module contains logging functionality."""
from __future__ import absolute_import

import logging


def configure_logger():
    """Add a handler to the library's logger with a formatter that
    includes a timestamp along with the message.
    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    get_logger().addHandler(handler)


def get_logger():
    """Return a logger with the name "sagemaker-inference",
    creating it if necessary.

    Returns:
        logging.Logger: Instance of the Logger class.
    """
    return logging.getLogger("sagemaker-inference")
