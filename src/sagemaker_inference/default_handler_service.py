# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

from sagemaker_inference.transformer import Transformer


class DefaultHandlerService(object):
    """Default handler service that is executed by the model server.

    The handler service is responsible for defining an ``initialize`` and ``handle`` method.
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.

    Implementation of: https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md
    """
    def __init__(self, transformer=None):
        self._service = transformer if transformer else Transformer()

    def handle(self, data, context):
        return self._service.transform(data, context)

    def initialize(self):
        self._service.validate_and_initialize()
