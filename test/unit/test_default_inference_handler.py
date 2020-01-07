# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from mock import patch
import pytest

from sagemaker_inference import content_types
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler


@patch('sagemaker_inference.decoder.decode')
def test_default_input_fn(loads):
    assert DefaultInferenceHandler().default_input_fn(42, content_types.JSON)

    loads.assert_called_with(42, content_types.JSON)


@patch('sagemaker_inference.encoder.encode', lambda prediction, accept: prediction ** 2)
def test_default_output_fn():
    result, accept = DefaultInferenceHandler().default_output_fn(2, content_types.CSV)
    assert result == 4
    assert accept == content_types.CSV


def test_default_model_fn():
    with pytest.raises(NotImplementedError):
        DefaultInferenceHandler().default_model_fn('model_dir')


def test_predict_fn():
    with pytest.raises(NotImplementedError):
        DefaultInferenceHandler().default_predict_fn('data', 'model')
