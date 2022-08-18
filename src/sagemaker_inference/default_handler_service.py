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
"""This module contains functionality for the default handler service."""
from __future__ import absolute_import

import os

from sagemaker_inference.transformer import Transformer

PYTHON_PATH_ENV = "PYTHONPATH"


class DefaultHandlerService(object):
    """Default handler service that is executed by the model server.

    The handler service is responsible for defining an ``initialize`` and ``handle`` method.
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.

    Implementation of: https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md
    """

    def __init__(self, transformer=None):
        self._service = transformer if transformer else Transformer()

    def handle(self, data, context):
        """Handles an inference request with input data and makes a prediction.

        Args:
            data (obj): the request data.
            context (obj): metadata on the incoming request data.

        Returns:
            list[obj]: The return value from the Transformer.transform method,
                which is a serialized prediction result wrapped in a list if
                inference is successful. Otherwise returns an error message
                with the context set appropriately.

        """
        return self._service.transform(data, context)

    def initialize(self, context):
        """Calls the Transformer method that validates the user module against
        the SageMaker inference contract.
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # add model_dir/code to python path
        code_dir_path = "{}:".format(model_dir + "/code")
        if PYTHON_PATH_ENV in os.environ:
            os.environ[PYTHON_PATH_ENV] = code_dir_path + os.environ[PYTHON_PATH_ENV]
        else:
            os.environ[PYTHON_PATH_ENV] = code_dir_path

        self._service.validate_and_initialize(model_dir=model_dir, context=context)
