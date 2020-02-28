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

import pkg_resources
import psutil
from retrying import retry

import sagemaker_inference
from sagemaker_inference import default_handler_service, environment, logging, utils
from sagemaker_inference.environment import code_dir

logger = logging.get_logger()

MMS_CONFIG_FILE = os.path.join("/etc", "sagemaker-mms.properties")
DEFAULT_HANDLER_SERVICE = default_handler_service.__name__
DEFAULT_MMS_CONFIG_FILE = pkg_resources.resource_filename(
    sagemaker_inference.__name__, "/etc/default-mms.properties"
)
MME_MMS_CONFIG_FILE = pkg_resources.resource_filename(
    sagemaker_inference.__name__, "/etc/mme-mms.properties"
)
DEFAULT_MMS_LOG_FILE = pkg_resources.resource_filename(
    sagemaker_inference.__name__, "/etc/log4j.properties"
)
DEFAULT_MMS_MODEL_DIRECTORY = os.path.join(os.getcwd(), ".sagemaker/mms/models")
DEFAULT_MMS_MODEL_NAME = "model"

ENABLE_MULTI_MODEL = os.getenv("SAGEMAKER_MULTI_MODEL", "false") == "true"
MODEL_STORE = "/" if ENABLE_MULTI_MODEL else DEFAULT_MMS_MODEL_DIRECTORY

PYTHON_PATH_ENV = "PYTHONPATH"
REQUIREMENTS_PATH = os.path.join(code_dir, "requirements.txt")
MMS_NAMESPACE = "com.amazonaws.ml.mms.ModelServer"


def start_model_server(handler_service=DEFAULT_HANDLER_SERVICE):
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
        _set_python_path()
    else:
        _adapt_to_mms_format(handler_service)

    _create_model_server_config_file()

    if os.path.exists(REQUIREMENTS_PATH):
        _install_requirements()

    mxnet_model_server_cmd = [
        "mxnet-model-server",
        "--start",
        "--model-store",
        MODEL_STORE,
        "--mms-config",
        MMS_CONFIG_FILE,
        "--log-config",
        DEFAULT_MMS_LOG_FILE,
    ]

    logger.info(mxnet_model_server_cmd)
    subprocess.Popen(mxnet_model_server_cmd)

    mms_process = _retrieve_mms_server_process()

    _add_sigterm_handler(mms_process)

    mms_process.wait()


def _adapt_to_mms_format(handler_service):
    if not os.path.exists(DEFAULT_MMS_MODEL_DIRECTORY):
        os.makedirs(DEFAULT_MMS_MODEL_DIRECTORY)

    model_archiver_cmd = [
        "model-archiver",
        "--model-name",
        DEFAULT_MMS_MODEL_NAME,
        "--handler",
        handler_service,
        "--model-path",
        environment.model_dir,
        "--export-path",
        DEFAULT_MMS_MODEL_DIRECTORY,
        "--archive-format",
        "no-archive",
    ]

    logger.info(model_archiver_cmd)
    subprocess.check_call(model_archiver_cmd)

    _set_python_path()


def _set_python_path():
    # MMS handles code execution by appending the export path, provided
    # to the model archiver, to the PYTHONPATH env var.
    # The code_dir has to be added to the PYTHONPATH otherwise the
    # user provided module can not be imported properly.
    code_dir_path = "{}:".format(environment.code_dir)

    if PYTHON_PATH_ENV in os.environ:
        os.environ[PYTHON_PATH_ENV] = code_dir_path + os.environ[PYTHON_PATH_ENV]
    else:
        os.environ[PYTHON_PATH_ENV] = code_dir_path


def _create_model_server_config_file():
    configuration_properties = _generate_mms_config_properties()

    utils.write_file(MMS_CONFIG_FILE, configuration_properties)


def _generate_mms_config_properties():
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
        default_configuration = utils.read_file(MME_MMS_CONFIG_FILE)
    else:
        default_configuration = utils.read_file(DEFAULT_MMS_CONFIG_FILE)

    return default_configuration + custom_configuration


def _add_sigterm_handler(mms_process):
    def _terminate(signo, frame):  # pylint: disable=unused-argument
        try:
            os.kill(mms_process.pid, signal.SIGTERM)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _terminate)


def _install_requirements():
    logger.info("installing packages from requirements.txt...")
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH]

    try:
        subprocess.check_call(pip_install_cmd)
    except subprocess.CalledProcessError:
        logger.error("failed to install required packages, exiting")
        raise ValueError("failed to install required packages")


# retry for 10 seconds
@retry(stop_max_delay=10 * 1000)
def _retrieve_mms_server_process():
    mms_server_processes = list()

    for process in psutil.process_iter():
        if MMS_NAMESPACE in process.cmdline():
            mms_server_processes.append(process)

    if not mms_server_processes:
        raise Exception("mms model server was unsuccessfully started")

    if len(mms_server_processes) > 1:
        raise Exception("multiple mms model servers are not supported")

    return mms_server_processes[0]
