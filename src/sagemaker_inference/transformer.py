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
"""This module contains functionality for the Transformer class,
which represents the execution workflow for handling inference
requests.
"""
from __future__ import absolute_import

import importlib
import traceback

try:
    from inspect import signature  # pylint: disable=ungrouped-imports
except ImportError:
    # for Python2.7
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "inspect2"])
    from inspect2 import signature

try:
    from importlib.util import find_spec  # pylint: disable=ungrouped-imports
except ImportError:
    import imp  # noqa: F401

    def find_spec(module_name):
        """Function that searches for a module.

        Args:
            module_name: The name of the module to search for.

        Returns:
            bool: Whether the module was found.
        """
        try:
            imp.find_module(module_name)
            return True
        except ImportError:
            return None


from six.moves import http_client

from sagemaker_inference import content_types, environment, utils
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
from sagemaker_inference.errors import BaseInferenceToolkitError, GenericInferenceToolkitError


class Transformer(object):
    """Represents the execution workflow for handling inference requests
    sent to the model server.
    """

    def __init__(self, default_inference_handler=None):
        """Initialize a ``Transformer``.

        Args:
            default_inference_handler (DefaultInferenceHandler): default implementation of
                inference handlers to use in absence of expected serving functions within
                the user module. Defaults to ``DefaultInferenceHandler``.

        """
        self._default_inference_handler = default_inference_handler or DefaultInferenceHandler()
        self._initialized = False
        self._environment = None
        self._model = None

        self._pre_model_fn = None
        self._model_warmup_fn = None
        self._model_fn = None
        self._transform_fn = None
        self._input_fn = None
        self._predict_fn = None
        self._output_fn = None
        self._context = None

    @staticmethod
    def handle_error(context, inference_exception, trace):
        """Set context appropriately for error response.

        Args:
            context (mms.context.Context): The inference context.
            inference_exception (sagemaker_inference.errors.BaseInferenceToolkitError): An exception
                raised during inference, with information for the error response.
            trace (traceback): The stacktrace of the error.

        Returns:
            str: The error message and stacktrace from the exception.
        """
        context.set_response_status(
            code=inference_exception.status_code,
            phrase=utils.remove_crlf(inference_exception.phrase),
        )
        return ["{}\n{}".format(inference_exception.message, trace)]

    def transform(self, data, context):
        """Take a request with input data, deserialize it, make a prediction, and return a
        serialized response.

        Args:
            data (obj): the request data.
            context (obj): metadata on the incoming request data.

        Returns:
            list[obj]: The serialized prediction result wrapped in a list if
                inference is successful. Otherwise returns an error message
                with the context set appropriately.
        """
        try:
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            self.validate_and_initialize(model_dir=model_dir, context=context)

            response_list = []

            for i in range(len(data)):
                input_data = data[i].get("body")

                request_processor = context.request_processor[0]

                request_property = request_processor.get_request_properties()
                content_type = utils.retrieve_content_type_header(request_property)
                accept = request_property.get("Accept") or request_property.get("accept")

                if not accept or accept == content_types.ANY:
                    accept = self._environment.default_accept

                if content_type in content_types.UTF8_TYPES:
                    input_data = input_data.decode("utf-8")

                result = self._run_handler_function(
                    self._transform_fn, *(self._model, input_data, content_type, accept)
                )

                response = result
                response_content_type = accept

                if isinstance(result, tuple):
                    # handles tuple for backwards compatibility
                    response = result[0]
                    response_content_type = result[1]

                context.set_response_content_type(0, response_content_type)

                response_list.append(response)

            return response_list
        except Exception as e:  # pylint: disable=broad-except
            trace = traceback.format_exc()
            if isinstance(e, BaseInferenceToolkitError):
                return self.handle_error(context, e, trace)
            else:
                return self.handle_error(
                    context,
                    GenericInferenceToolkitError(http_client.INTERNAL_SERVER_ERROR, str(e)),
                    trace,
                )

    def validate_and_initialize(self, model_dir=environment.model_dir, context=None):
        """Validates the user module against the SageMaker inference contract.

        Load the model as defined by the ``model_fn`` to prepare handling predictions.

        """
        if not self._initialized:
            self._context = context
            self._environment = environment.Environment()
            self._validate_user_module_and_set_functions()

            if self._pre_model_fn is not None:
                self._run_handler_function(self._pre_model_fn, *(model_dir,))

            self._model = self._run_handler_function(self._model_fn, *(model_dir,))

            if self._model_warmup_fn is not None:
                self._run_handler_function(self._model_warmup_fn, *(model_dir, self._model))

            self._initialized = True

    def _validate_user_module_and_set_functions(self):
        """Retrieves and validates the inference handlers provided within the user module.

        Default implementations of the inference handlers are utilized in
        place of missing functions defined in the user module.

        """
        user_module_name = self._environment.module_name

        self._pre_model_fn = getattr(self._default_inference_handler, "default_pre_model_fn", None)
        self._model_warmup_fn = getattr(
            self._default_inference_handler, "default_model_warmup_fn", None
        )

        if find_spec(user_module_name) is not None:
            user_module = importlib.import_module(user_module_name)

            self._model_fn = getattr(
                user_module, "model_fn", self._default_inference_handler.default_model_fn
            )

            transform_fn = getattr(user_module, "transform_fn", None)
            input_fn = getattr(user_module, "input_fn", None)
            predict_fn = getattr(user_module, "predict_fn", None)
            output_fn = getattr(user_module, "output_fn", None)
            pre_model_fn = getattr(user_module, "pre_model_fn", None)
            model_warmup_fn = getattr(user_module, "model_warmup_fn", None)

            if transform_fn and (input_fn or predict_fn or output_fn):
                raise ValueError(
                    "Cannot use transform_fn implementation in conjunction with "
                    "input_fn, predict_fn, and/or output_fn implementation"
                )

            self._transform_fn = transform_fn or self._default_transform_fn
            self._input_fn = input_fn or self._default_inference_handler.default_input_fn
            self._predict_fn = predict_fn or self._default_inference_handler.default_predict_fn
            self._output_fn = output_fn or self._default_inference_handler.default_output_fn
            if pre_model_fn is not None:
                self._pre_model_fn = pre_model_fn
            if model_warmup_fn is not None:
                self._model_warmup_fn = model_warmup_fn
        else:
            self._model_fn = self._default_inference_handler.default_model_fn
            self._input_fn = self._default_inference_handler.default_input_fn
            self._predict_fn = self._default_inference_handler.default_predict_fn
            self._output_fn = self._default_inference_handler.default_output_fn

            self._transform_fn = self._default_transform_fn

    def _default_transform_fn(self, model, input_data, content_type, accept, context=None):
        # pylint: disable=unused-argument
        """Make predictions against the model and return a serialized response.
        This serves as the default implementation of transform_fn, used when the
        user has not provided an implementation.

        Args:
            model (obj): model loaded by model_fn.
            input_data (obj): the request data.
            content_type (str): the request content type.
            accept (str): accept header expected by the client.
            context (obj): the request context (default: None).

        Returns:
            obj: the serialized prediction result or a tuple of the form
                (response_data, content_type)

        """
        data = self._run_handler_function(self._input_fn, *(input_data, content_type))
        prediction = self._run_handler_function(self._predict_fn, *(data, model))
        result = self._run_handler_function(self._output_fn, *(prediction, accept))
        return result

    def _run_handler_function(self, func, *argv):
        """Helper to call the handler function which covers 2 cases:
        1. the handle function takes context
        2. the handle function does not take context
        """
        num_func_input = len(signature(func).parameters)
        if num_func_input == len(argv):
            # function does not take context
            result = func(*argv)
        elif num_func_input == len(argv) + 1:
            # function takes context
            argv_context = argv + (self._context,)
            result = func(*argv_context)
        else:
            raise TypeError(
                "{} takes {} arguments but {} were given.".format(
                    func.__name__, num_func_input, len(argv)
                )
            )

        return result
