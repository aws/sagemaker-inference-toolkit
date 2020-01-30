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
"""This module contains custom exceptions."""
from __future__ import absolute_import

import textwrap


class UnsupportedFormatError(Exception):
    """Exception used to indicate that an unsupported content type was provided."""

    def __init__(self, content_type, **kwargs):
        self._message = textwrap.dedent(
            """Content type %s is not supported by this framework.

            Please implement input_fn to to deserialize the request data or an output_fn to
            serialize the response. For more information, see the SageMaker Python SDK README."""
            % content_type
        )
        super(UnsupportedFormatError, self).__init__(self._message, **kwargs)


class BaseInferenceToolkitError(Exception):
    """Exception used to indicate a problem that occurred during inference.

    This is meant to be extended from so that customers may handle errors
    within inference servers.
    """

    def __init__(self, status_code, message, phrase):
        """Initializes an instance of BaseInferenceToolkitError.

        Args:
            status_code (int): HTTP Error Status Code to send to client
            message (str): Response message to send to client
            phrase (str): Response body to send to client
        """
        self.status_code = status_code
        self.message = message
        self.phrase = phrase
        super(BaseInferenceToolkitError, self).__init__(status_code, message, phrase)


class GenericInferenceToolkitError(BaseInferenceToolkitError):
    """Exception used to indicate a problem that occurred during inference.

    This is meant to be a generic implementation of the BaseInferenceToolkitError
    for re-raising unexpected exceptions in a way that can be sent back to the client.
    """

    def __init__(self, status_code, message=None, phrase=None):
        """Initializes an instance of GenericInferenceToolkitError.

        Args:
            status_code (int): HTTP Error Status Code to send to client
            message (str): Response message to send to client
            phrase (str): Response body to send to client
        """
        message = message or "Invalid Request"
        phrase = phrase or message
        super(GenericInferenceToolkitError, self).__init__(status_code, message, phrase)
