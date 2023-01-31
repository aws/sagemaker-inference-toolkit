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
from mock import MagicMock, Mock, patch

from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer

DATA = "data"
CONTEXT = Mock()
TRANSFORMED_RESULT = "transformed_result"


@patch("importlib.import_module", return_value=object())
def test_default_handler_service(import_lib):
    handler_service = DefaultHandlerService()

    assert isinstance(handler_service._service, Transformer)


def test_default_handler_service_custom_transformer():
    transformer = Mock()

    handler_service = DefaultHandlerService(transformer)

    assert handler_service._service == transformer


def test_handle():
    transformer = Mock()
    transformer.transform.return_value = TRANSFORMED_RESULT

    handler_service = DefaultHandlerService(transformer)
    result = handler_service.handle(DATA, CONTEXT)

    assert result == TRANSFORMED_RESULT
    transformer.transform.assert_called_once_with(DATA, CONTEXT)


def test_initialize():
    transformer = Mock()
    properties = {"model_dir": "/opt/ml/models/model-name"}

    def getitem(key):
        return properties[key]

    context = MagicMock()
    context.system_properties.__getitem__.side_effect = getitem
    DefaultHandlerService(transformer).initialize(context)

    transformer.validate_and_initialize.assert_called_once()
