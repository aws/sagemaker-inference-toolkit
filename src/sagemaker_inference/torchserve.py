# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This module contains functionality to configure and start the
multi-model server."""
from __future__ import absolute_import

import os
import signal
import subprocess
import sys
import importlib

import pkg_resources
import psutil
from retrying import retry

import sagemaker_inference
from sagemaker_inference import default_handler_service, environment, logging, utils
from sagemaker_inference.model_server_utils import add_sigterm_handler, set_python_path, install_requirements, retrieve_model_server_process
from sagemaker_inference.environment import code_dir

logger = logging.get_logger()

TS_CONFIG_FILE = os.path.join("/etc", "sagemaker-ts.properties")
DEFAULT_TS_CONFIG_FILE = pkg_resources.resource_filename(
    sagemaker_inference.__name__, "/etc/default-ts.properties"
)
MME_TS_CONFIG_FILE = pkg_resources.resource_filename(
    sagemaker_inference.__name__, "/etc/mme-ts.properties"
)
DEFAULT_TS_LOG_FILE = pkg_resources.resource_filename(
    sagemaker_inference.__name__, "/etc/ts.log4j.properties"
)
DEFAULT_TS_MODEL_DIRECTORY = os.path.join(os.getcwd(), ".sagemaker/ts/models")
DEFAULT_TS_MODEL_NAME = "model"
DEFAULT_TS_MODEL_SERIALIZED_FILE = "model.pth"
DEFAULT_TS_HANDLER_SERVICE = "sagemaker_pytorch_serving_container.handler_service"

ENABLE_MULTI_MODEL = os.getenv("SAGEMAKER_MULTI_MODEL", "false") == "true"
MODEL_STORE = "/" if ENABLE_MULTI_MODEL else DEFAULT_TS_MODEL_DIRECTORY

PYTHON_PATH_ENV = "PYTHONPATH"
REQUIREMENTS_PATH = os.path.join(code_dir, "requirements.txt")
TS_NAMESPACE = "org.pytorch.serve.ModelServer"


def start_model_server(handler_service=DEFAULT_TS_HANDLER_SERVICE):
    """Configure and start the model server.

    Args:
        handler_service (str): python path pointing to a module that defines
            a class with the following:

                - A ``handle`` method, which is invoked for all incoming inference
                    requests to the model server.
                - A ``initialize`` method, which is invoked at model server start up
                    for loading the model.

            Defaults to ``sagemaker_inference.default_handler_service``.

    """

    if ENABLE_MULTI_MODEL:
        if not os.getenv("SAGEMAKER_HANDLER"):
            os.environ["SAGEMAKER_HANDLER"] = handler_service
        set_python_path()
    else:
        _adapt_to_ts_format(handler_service)

    _create_torchserve_config_file()

    if os.path.exists(REQUIREMENTS_PATH):
        install_requirements()

    ts_model_server_cmd = [
        "torchserve",
        "--start",
        "--model-store",
        MODEL_STORE,
        "--ts-config",
        TS_CONFIG_FILE,
        "--log-config",
        DEFAULT_TS_LOG_FILE,
        "--models",
        "model.mar"
    ]

    logger.info(ts_model_server_cmd)
    subprocess.Popen(ts_model_server_cmd)

    ts_process = retrieve_model_server_process(TS_NAMESPACE)

    add_sigterm_handler(ts_process)

    ts_process.wait()


def _adapt_to_ts_format(handler_service):
    if not os.path.exists(DEFAULT_TS_MODEL_DIRECTORY):
        os.makedirs(DEFAULT_TS_MODEL_DIRECTORY)


    model_archiver_cmd = [
        "torch-model-archiver",
        "--model-name",
        DEFAULT_TS_MODEL_NAME,
        "--handler",
        handler_service,
        "--serialized-file",
        os.path.join(environment.model_dir, DEFAULT_TS_MODEL_SERIALIZED_FILE),
        "--export-path",
        DEFAULT_TS_MODEL_DIRECTORY,
        "--extra-files",
        os.path.join(environment.model_dir, environment.Environment().module_name + ".py"),
        "--version",
        "1",
    ]

    logger.info(model_archiver_cmd)
    subprocess.check_call(model_archiver_cmd)

    set_python_path()


def _create_torchserve_config_file():
    configuration_properties = _generate_ts_config_properties()

    utils.write_file(TS_CONFIG_FILE, configuration_properties)


def _generate_ts_config_properties():
    env = environment.Environment()

    user_defined_configuration = {
        "default_response_timeout": env.model_server_timeout,
        "default_workers_per_model": env.model_server_workers,
        "inference_address": "http://0.0.0.0:{}".format(env.inference_http_port),
        "management_address": "http://0.0.0.0:{}".format(env.management_http_port),
    }

    custom_configuration = str()

    for key in user_defined_configuration:
        value = user_defined_configuration.get(key)
        if value:
            custom_configuration += "{}={}\n".format(key, value)

    if ENABLE_MULTI_MODEL:
        default_configuration = utils.read_file(MME_TS_CONFIG_FILE)
    else:
        default_configuration = utils.read_file(DEFAULT_TS_CONFIG_FILE)

    return default_configuration + custom_configuration
