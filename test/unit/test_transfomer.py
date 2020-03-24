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
import pytest

try:
    import http.client as http_client
except ImportError:
    import httplib as http_client

from sagemaker_inference import content_types, environment
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
from sagemaker_inference.errors import BaseInferenceToolkitError
from sagemaker_inference.transformer import Transformer

INPUT_DATA = "input_data"
CONTENT_TYPE = "content_type"
ACCEPT = "accept"
DEFAULT_ACCEPT = "default_accept"
RESULT = "result"
MODEL = "foo"

PREPROCESSED_DATA = "preprocessed_data"
PREDICT_RESULT = "prediction_result"
PROCESSED_RESULT = "processed_result"


def test_default_transformer():
    transformer = Transformer()

    assert isinstance(transformer._default_inference_handler, DefaultInferenceHandler)
    assert len(transformer._initialized) == 0
    assert transformer._environment is None
    assert transformer._model is None
    assert transformer._model_fn is None
    assert transformer._transform_fn is None
    assert transformer._input_fn is None
    assert transformer._predict_fn is None
    assert transformer._output_fn is None


def test_transformer_with_custom_default_inference_handler():
    default_inference_handler = Mock()

    transformer = Transformer(default_inference_handler)

    assert transformer._default_inference_handler == default_inference_handler
    assert len(transformer._initialized) == 0
    assert transformer._environment is None
    assert transformer._model is None
    assert transformer._model_fn is None
    assert transformer._transform_fn is None
    assert transformer._input_fn is None
    assert transformer._predict_fn is None
    assert transformer._output_fn is None


@pytest.mark.parametrize("accept_key", ["Accept", "accept"])
@patch("sagemaker_inference.utils.retrieve_content_type_header", return_value=CONTENT_TYPE)
@patch("sagemaker_inference.transformer.Transformer.validate_and_initialize")
def test_transform(validate, retrieve_content_type_header, accept_key):
    data = [{"body": INPUT_DATA}]
    context = Mock()
    request_processor = Mock()
    transform_fn = Mock(return_value=RESULT)

    context.request_processor = [request_processor]
    request_property = {accept_key: ACCEPT}
    request_processor.get_request_properties.return_value = request_property

    transformer = Transformer()
    transformer._model = MODEL
    transformer._transform_fn = transform_fn

    result = transformer.transform(data, context)

    validate.assert_called_once()
    retrieve_content_type_header.assert_called_once_with(request_property)
    transform_fn.assert_called_once_with(MODEL, INPUT_DATA, CONTENT_TYPE, ACCEPT)
    context.set_response_content_type.assert_called_once_with(0, ACCEPT)
    assert isinstance(result, list)
    assert result[0] == RESULT


@patch("sagemaker_inference.utils.retrieve_content_type_header", return_value=CONTENT_TYPE)
@patch("sagemaker_inference.transformer.Transformer.validate_and_initialize")
def test_transform_no_accept(validate, retrieve_content_type_header):
    data = [{"body": INPUT_DATA}]
    context = Mock()
    request_processor = Mock()
    transform_fn = Mock()
    environment = Mock()
    environment.default_accept = DEFAULT_ACCEPT

    context.request_processor = [request_processor]
    request_processor.get_request_properties.return_value = dict()

    transformer = Transformer()
    transformer._model = MODEL
    transformer._transform_fn = transform_fn
    transformer._environment = environment

    transformer.transform(data, context)

    validate.assert_called_once()
    transform_fn.assert_called_once_with(MODEL, INPUT_DATA, CONTENT_TYPE, DEFAULT_ACCEPT)


@patch("sagemaker_inference.utils.retrieve_content_type_header", return_value=CONTENT_TYPE)
@patch("sagemaker_inference.transformer.Transformer.validate_and_initialize")
def test_transform_any_accept(validate, retrieve_content_type_header):
    data = [{"body": INPUT_DATA}]
    context = Mock()
    request_processor = Mock()
    transform_fn = Mock()
    environment = Mock()
    environment.default_accept = DEFAULT_ACCEPT

    context.request_processor = [request_processor]
    request_processor.get_request_properties.return_value = {"accept": content_types.ANY}

    transformer = Transformer()
    transformer._model = MODEL
    transformer._transform_fn = transform_fn
    transformer._environment = environment

    transformer.transform(data, context)

    validate.assert_called_once()
    transform_fn.assert_called_once_with(MODEL, INPUT_DATA, CONTENT_TYPE, DEFAULT_ACCEPT)


@pytest.mark.parametrize("content_type", content_types.UTF8_TYPES)
@patch("sagemaker_inference.utils.retrieve_content_type_header")
@patch("sagemaker_inference.transformer.Transformer.validate_and_initialize")
def test_transform_decode(validate, retrieve_content_type_header, content_type):
    input_data = Mock()
    context = Mock()
    request_processor = Mock()
    transform_fn = Mock()
    data = [{"body": input_data}]

    input_data.decode.return_value = INPUT_DATA
    context.request_processor = [request_processor]
    request_processor.get_request_properties.return_value = {"accept": ACCEPT}
    retrieve_content_type_header.return_value = content_type

    transformer = Transformer()
    transformer._model = MODEL
    transformer._transform_fn = transform_fn

    transformer.transform(data, context)

    input_data.decode.assert_called_once_with("utf-8")
    transform_fn.assert_called_once_with(MODEL, INPUT_DATA, content_type, ACCEPT)


@patch("sagemaker_inference.utils.retrieve_content_type_header", return_value=CONTENT_TYPE)
@patch("sagemaker_inference.transformer.Transformer.validate_and_initialize")
def test_transform_tuple(validate, retrieve_content_type_header):
    data = [{"body": INPUT_DATA}]
    context = Mock()
    request_processor = Mock()
    transform_fn = Mock(return_value=(RESULT, ACCEPT))

    context.request_processor = [request_processor]
    request_processor.get_request_properties.return_value = {"accept": ACCEPT}

    transformer = Transformer()
    transformer._model = MODEL
    transformer._transform_fn = transform_fn

    result = transformer.transform(data, context)

    transform_fn.assert_called_once_with(MODEL, INPUT_DATA, CONTENT_TYPE, ACCEPT)
    context.set_response_content_type.assert_called_once_with(0, transform_fn()[1])
    assert isinstance(result, list)
    assert result[0] == transform_fn()[0]


@patch("sagemaker_inference.transformer.Transformer._validate_inference_handlers_and_set_functions")
@patch("sagemaker_inference.environment.Environment")
def test_validate_and_initialize(env, validate_user_module):
    transformer = Transformer()

    model_fn = Mock()
    transformer._model_fn = model_fn

    assert len(transformer._initialized) == 0

    transformer.validate_and_initialize()

    assert len(transformer._initialized) == 1

    transformer.validate_and_initialize()

    model_fn.assert_called_once_with(environment.model_dir)
    env.assert_called_once_with()
    validate_user_module.assert_called_once_with()


@patch("sagemaker_inference.transformer.Transformer._validate_inference_handlers_and_set_functions")
@patch("sagemaker_inference.environment.Environment")
@patch("os.path.join")
def test_handle_validate_and_initialize_error(env, validate_user_module, os_path_join):
    data = [{"body": INPUT_DATA}]
    request_processor = Mock()

    context = Mock()
    context.request_processor = [request_processor]

    transform_fn = Mock()
    model_fn = Mock()

    transformer = Transformer()

    transformer._model = MODEL
    transformer._transform_fn = transform_fn
    transformer._model_fn = model_fn

    test_error_message = "Foo"
    validate_user_module.side_effect = ValueError(test_error_message)

    assert len(transformer._initialized) == 0

    response = transformer.transform(data, context)
    assert test_error_message in str(response)
    context.set_response_status.assert_called_with(
        code=http_client.INTERNAL_SERVER_ERROR, phrase=test_error_message
    )


@patch("sagemaker_inference.transformer.Transformer._validate_inference_handlers_and_set_functions")
@patch("sagemaker_inference.environment.Environment")
@patch("os.path.join")
def test_handle_validate_and_initialize_user_error(env, validate_user_module, os_path_join):
    test_status_code = http_client.FORBIDDEN
    test_error_message = "Foo"

    class FooUserError(BaseInferenceToolkitError):
        def __init__(self, status_code, message):
            self.status_code = status_code
            self.message = message
            self.phrase = "Foo"

    data = [{"body": INPUT_DATA}]
    context = Mock()
    transform_fn = Mock()
    model_fn = Mock()

    transformer = Transformer()

    transformer._model = MODEL
    transformer._transform_fn = transform_fn
    transformer._model_fn = model_fn

    validate_user_module.side_effect = FooUserError(test_status_code, test_error_message)

    assert len(transformer._initialized) == 0

    response = transformer.transform(data, context)
    assert test_error_message in str(response)
    context.set_response_status.assert_called_with(
        code=http_client.FORBIDDEN, phrase=test_error_message
    )


class UserModuleMock:
    def __init__(self, transform_fn=Mock(), input_fn=Mock(), predict_fn=Mock(), output_fn=Mock()):
        self.transform_fn = transform_fn
        self.input_fn = input_fn
        self.predict_fn = predict_fn
        self.output_fn = output_fn


@patch("importlib.import_module")
@patch("sagemaker_inference.transformer.find_spec", return_value=None)
def test_validate_no_user_module_and_set_functions(find_spec, import_module):
    default_inference_handler = Mock()
    mock_env = Mock()
    mock_env.module_name = "foo_module"

    default_model_fn = object()
    default_input_fn = object()
    default_predict_fn = object()
    default_output_fn = object()

    default_inference_handler.default_model_fn = default_model_fn
    default_inference_handler.default_input_fn = default_input_fn
    default_inference_handler.default_predict_fn = default_predict_fn
    default_inference_handler.default_output_fn = default_output_fn

    transformer = Transformer(default_inference_handler)
    transformer._environment = mock_env
    transformer._validate_inference_handlers_and_set_functions()

    find_spec.assert_called_once_with(mock_env.module_name)
    import_module.assert_not_called()
    assert transformer._default_inference_handler == default_inference_handler
    assert transformer._environment == mock_env
    assert transformer._model_fn == default_model_fn
    assert transformer._input_fn == default_input_fn
    assert transformer._predict_fn == default_predict_fn
    assert transformer._output_fn == default_output_fn


@patch("importlib.import_module", return_value=object())
@patch("sagemaker_inference.transformer.find_spec", return_value=Mock())
def test_validate_user_module_and_set_functions(find_spec, import_module):
    default_inference_handler = Mock()
    mock_env = Mock()
    mock_env.module_name = "foo_module"

    default_model_fn = object()
    default_input_fn = object()
    default_predict_fn = object()
    default_output_fn = object()

    default_inference_handler.default_model_fn = default_model_fn
    default_inference_handler.default_input_fn = default_input_fn
    default_inference_handler.default_predict_fn = default_predict_fn
    default_inference_handler.default_output_fn = default_output_fn

    transformer = Transformer(default_inference_handler)
    transformer._environment = mock_env
    transformer._validate_inference_handlers_and_set_functions()

    find_spec.assert_called_once_with(mock_env.module_name)
    import_module.assert_called_once_with(mock_env.module_name)
    assert transformer._default_inference_handler == default_inference_handler
    assert transformer._environment == mock_env
    assert transformer._model_fn == default_model_fn
    assert transformer._input_fn == default_input_fn
    assert transformer._predict_fn == default_predict_fn
    assert transformer._output_fn == default_output_fn


@patch(
    "importlib.import_module",
    return_value=UserModuleMock(input_fn=None, predict_fn=None, output_fn=None),
)
@patch("sagemaker_inference.transformer.find_spec", return_value=Mock())
def test_validate_user_module_and_set_functions_transform_fn(find_spec, import_module):
    mock_env = Mock()
    mock_env.module_name = "foo_module"

    import_module.transform_fn = Mock()

    transformer = Transformer()
    transformer._environment = mock_env

    transformer._validate_inference_handlers_and_set_functions()

    find_spec.assert_called_once_with(mock_env.module_name)
    import_module.assert_called_once_with(mock_env.module_name)
    assert transformer._transform_fn == import_module.return_value.transform_fn


def _assert_value_error_raised():
    with pytest.raises(ValueError) as e:
        transformer = Transformer()
        transformer._environment = Mock()
        transformer._validate_inference_handlers_and_set_functions()

    assert (
        "Cannot use transform_fn implementation in conjunction with input_fn, predict_fn, "
        "and/or output_fn implementation" in str(e.value)
    )


@pytest.mark.parametrize(
    "user_module",
    [
        UserModuleMock(input_fn=None),
        UserModuleMock(predict_fn=None),
        UserModuleMock(output_fn=None),
        UserModuleMock(output_fn=None, predict_fn=None),
        UserModuleMock(input_fn=None, output_fn=None),
        UserModuleMock(input_fn=None, predict_fn=None),
        UserModuleMock(),
    ],
)
@patch("importlib.import_module")
@patch("sagemaker_inference.transformer.find_spec", return_value=Mock())
def test_validate_user_module_error(find_spec, import_module, user_module):
    import_module.return_value = user_module

    _assert_value_error_raised()


def test_default_transform_fn():
    transformer = Transformer()

    input_fn = Mock(return_value=PREPROCESSED_DATA)
    predict_fn = Mock(return_value=PREDICT_RESULT)
    output_fn = Mock(return_value=PROCESSED_RESULT)

    transformer._input_fn = input_fn
    transformer._predict_fn = predict_fn
    transformer._output_fn = output_fn

    result = transformer._default_transform_fn(MODEL, INPUT_DATA, CONTENT_TYPE, ACCEPT)

    input_fn.assert_called_once_with(INPUT_DATA, CONTENT_TYPE)
    predict_fn.assert_called_once_with(PREPROCESSED_DATA, MODEL)
    output_fn.assert_called_once_with(PREDICT_RESULT, ACCEPT)
    assert result == PROCESSED_RESULT
