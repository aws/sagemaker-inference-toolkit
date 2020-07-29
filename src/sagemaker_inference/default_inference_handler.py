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
"""This module contains the definition of the default inference handler,
which provides a bare-bones implementation of default inference functions.
"""
import textwrap

from sagemaker_inference import decoder, encoder, errors, utils


class DefaultInferenceHandler(object):
    """Bare-bones implementation of default inference functions.
    """

    def default_model_fn(self, model_dir):
        """Function responsible for loading the model.

        Args:
            model_dir (str): The directory where model files are stored.

        Returns:
            obj: the loaded model.

        """
        raise NotImplementedError(
            textwrap.dedent(
                """
            Please provide a model_fn implementation.
            See documentation for model_fn at https://sagemaker.readthedocs.io/en/stable/
            """
            )
        )

    def default_input_fn(self, input_data, content_type):  # pylint: disable=no-self-use
        """Function responsible for deserializing the input data into an object for prediction.

        Args:
            input_data (obj): the request data.
            content_type (str): the request content type.

        Returns:
            obj: data ready for prediction.

        """
        return decoder.decode(input_data, content_type)

    def default_predict_fn(self, data, model):
        """Function responsible for model predictions.

        Args:
            model (obj): model loaded by the model_fn
            data: deserialized data returned by the input_fn

        Returns:
            obj: prediction result.

        """
        raise NotImplementedError(
            textwrap.dedent(
                """
            Please provide a predict_fn implementation.
            See documentation for predict_fn at https://sagemaker.readthedocs.io/en/stable/
            """
            )
        )

    def default_output_fn(self, prediction, accept):  # pylint: disable=no-self-use
        """Function responsible for serializing the prediction result to the desired accept type.

        Args:
            prediction (obj): prediction result returned by the predict_fn.
            accept (str): accept header expected by the client.

        Returns:
            obj: prediction data.

        """
        for content_type in utils.parse_accept(accept):
            if content_type in encoder.SUPPORTED_CONTENT_TYPES:
                return encoder.encode(prediction, content_type), content_type
        raise errors.UnsupportedFormatError(accept)
