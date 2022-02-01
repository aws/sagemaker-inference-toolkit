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
"""This module contains functionality that provides access to system
characteristics, environment variables and configuration settings.
"""
from __future__ import absolute_import

import os

from sagemaker_inference import content_types, logging, parameters

logger = logging.get_logger()

DEFAULT_MODULE_NAME = "inference.py"
DEFAULT_MODEL_SERVER_TIMEOUT = "60"
DEFAULT_STARTUP_TIMEOUT = "600"  # 10 minutes
DEFAULT_HTTP_PORT = "8080"

SAGEMAKER_BASE_PATH = os.path.join("/opt", "ml")  # type: str

base_dir = os.environ.get(parameters.BASE_PATH_ENV, SAGEMAKER_BASE_PATH)  # type: str

if os.environ.get(parameters.MULTI_MODEL_ENV) == "true":
    model_dir = os.path.join(base_dir, "models")  # type: str
else:
    model_dir = os.path.join(base_dir, "model")  # type: str
# str: the directory where models should be saved, e.g., /opt/ml/model/

code_dir = os.path.join(model_dir, "code")  # type: str
"""str: the path of the user's code directory, e.g., /opt/ml/model/code/"""


class Environment(object):
    """Provides access to aspects of the serving environment relevant to serving containers,
    including system characteristics, environment variables and configuration settings.

    The Environment is a read-only snapshot of the container environment.
    It is a dictionary-like object, allowing any builtin function that works with dictionary.

    Attributes:
        module_name (str): The name of the user-provided module. Default is inference.py.
        model_server_timeout (int): Timeout in seconds for the model server. Default is 60.
        model_server_workers (str): Number of worker processes the model server will use.

        default_accept (str): The desired default MIME type of the inference in the response
            as specified in the user-supplied SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT environment
            variable. Otherwise, returns 'application/json' by default.
            For example: application/json
        http_port (str): Port that SageMaker will use to handle invocations and pings against the
            running Docker container. Default is 8080. For example: 8080
        safe_port_range (str): HTTP port range that can be used by customers to avoid collisions
            with the HTTP port specified by SageMaker for handling pings and invocations.
            For example: 1111-2222

    """

    def __init__(self):
        self._module_name = os.environ.get(parameters.USER_PROGRAM_ENV, DEFAULT_MODULE_NAME)
        self._model_server_timeout = int(
            os.environ.get(parameters.MODEL_SERVER_TIMEOUT_ENV, DEFAULT_MODEL_SERVER_TIMEOUT)
        )
        self._model_server_workers = os.environ.get(parameters.MODEL_SERVER_WORKERS_ENV)
        self._startup_timeout = int(
            os.environ.get(parameters.STARTUP_TIMEOUT_ENV, DEFAULT_STARTUP_TIMEOUT)
        )
        self._default_accept = os.environ.get(
            parameters.DEFAULT_INVOCATIONS_ACCEPT_ENV, content_types.JSON
        )
        self._inference_http_port = os.environ.get(parameters.BIND_TO_PORT_ENV, DEFAULT_HTTP_PORT)
        self._management_http_port = os.environ.get(parameters.BIND_TO_PORT_ENV, DEFAULT_HTTP_PORT)
        self._safe_port_range = os.environ.get(parameters.SAFE_PORT_RANGE_ENV)

    @staticmethod
    def _parse_module_name(program_param):
        """Given a module name or a script name, return the module name.

        Args:
            program_param (str): Module or script name.

        Returns:
            str: Module name.

        """
        if program_param and program_param.endswith(".py"):
            return program_param[:-3]
        return program_param

    @property
    def module_name(self):  # type: () -> str
        """str: Name of the user-provided module."""
        return self._parse_module_name(self._module_name)

    @property
    def model_server_timeout(self):  # type: () -> int
        """int: Timeout, in seconds, used for model server's backend workers before
        they are deemed unresponsive and rebooted.
        """
        return self._model_server_timeout

    @property
    def model_server_workers(self):  # type: () -> str
        """str: Number of worker processes the model server is configured to use."""
        return self._model_server_workers

    @property
    def startup_timeout(self):  # type () -> int
        """int: Timeout, in seconds, used for starting up the model server and fetching
        its process id, before giving up and throwing error.
        """
        return self._startup_timeout

    @property
    def default_accept(self):  # type: () -> str
        """str: The desired default MIME type of the inference in the response."""
        return self._default_accept

    @property
    def inference_http_port(self):  # type: () -> str
        """str: HTTP port that SageMaker uses to handle invocations and pings."""
        return self._inference_http_port

    @property
    def management_http_port(self):  # type: () -> str
        """str: HTTP port that SageMaker uses to handle model management requests."""
        return self._management_http_port

    @property
    def safe_port_range(self):  # type: () -> str
        """str: HTTP port range that can be used by users to avoid collisions with the HTTP port
        specified by SageMaker for handling pings and invocations.
        """
        return self._safe_port_range
