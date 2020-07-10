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
import os
import signal
import subprocess
import types

from mock import Mock, patch
import pytest

from sagemaker_inference import environment, torchserve, model_server_utils
from sagemaker_inference.torchserve import TS_NAMESPACE, REQUIREMENTS_PATH

PYTHON_PATH = "python_path"
DEFAULT_CONFIGURATION = "default_configuration"


@patch("subprocess.call")
@patch("subprocess.Popen")
@patch("sagemaker_inference.torchserve.retrieve_model_server_process")
@patch("sagemaker_inference.torchserve.add_sigterm_handler")
@patch("sagemaker_inference.torchserve.install_requirements")
@patch("os.path.exists", return_value=True)
@patch("sagemaker_inference.torchserve._create_torchserve_config_file")
@patch("sagemaker_inference.torchserve._adapt_to_ts_format")
def test_start_torchserve_default_service_handler(
    adapt,
    create_config,
    exists,
    install_requirements,
    sigterm,
    retrieve,
    subprocess_popen,
    subprocess_call,
):
    torchserve.start_model_server()

    adapt.assert_called_once_with(torchserve.DEFAULT_TS_HANDLER_SERVICE)
    create_config.assert_called_once_with()
    exists.assert_called_once_with(REQUIREMENTS_PATH)
    install_requirements.assert_called_once_with()

    ts_model_server_cmd = [
        "torchserve",
        "--start",
        "--model-store",
        torchserve.MODEL_STORE,
        "--ts-config",
        torchserve.TS_CONFIG_FILE,
        "--log-config",
        torchserve.DEFAULT_TS_LOG_FILE,
        "--models",
        "model.mar"
    ]

    subprocess_popen.assert_called_once_with(ts_model_server_cmd)
    retrieve.assert_called_once_with(torchserve.TS_NAMESPACE)
    sigterm.assert_called_once_with(retrieve.return_value)


@patch("subprocess.call")
@patch("subprocess.Popen")
@patch("sagemaker_inference.torchserve.retrieve_model_server_process")
@patch("sagemaker_inference.torchserve.add_sigterm_handler")
@patch("sagemaker_inference.torchserve._create_torchserve_config_file")
@patch("sagemaker_inference.torchserve._adapt_to_ts_format")
def test_start_torchserve_custom_handler_service(
    adapt, create_config, sigterm, retrieve, subprocess_popen, subprocess_call
):
    handler_service = Mock()

    torchserve.start_model_server(handler_service)

    adapt.assert_called_once_with(handler_service)


@patch("sagemaker_inference.torchserve.set_python_path")
@patch("subprocess.check_call")
@patch("os.makedirs")
@patch("os.path.exists", return_value=False)
def test_adapt_to_ts_format(path_exists, make_dir, subprocess_check_call, set_python_path):
    handler_service = Mock()

    torchserve._adapt_to_ts_format(handler_service)

    path_exists.assert_called_once_with(torchserve.DEFAULT_TS_MODEL_DIRECTORY)
    make_dir.assert_called_once_with(torchserve.DEFAULT_TS_MODEL_DIRECTORY)

    model_archiver_cmd = [
        "torch-model-archiver",
        "--model-name",
        torchserve.DEFAULT_TS_MODEL_NAME,
        "--handler",
        handler_service,
        #importlib.import_module(DEFAULT_TS_HANDLER_SERVICE).__file__,
        "--serialized-file",
        os.path.join(environment.model_dir, torchserve.DEFAULT_TS_MODEL_SERIALIZED_FILE),
        "--export-path",
        torchserve.DEFAULT_TS_MODEL_DIRECTORY,
        "--extra-files",
        os.path.join(environment.model_dir, environment.Environment().module_name + ".py"),
        "--version",
        "1",
    ]

    subprocess_check_call.assert_called_once_with(model_archiver_cmd)
    subprocess_check_call.assert_called_once()
    set_python_path.assert_called_once_with()


@patch("sagemaker_inference.torchserve.set_python_path")
@patch("subprocess.check_call")
@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
def test_adapt_to_ts_format_existing_path(
    path_exists, make_dir, subprocess_check_call, set_python_path
):
    handler_service = Mock()

    torchserve._adapt_to_ts_format(handler_service)

    path_exists.assert_called_once_with(torchserve.DEFAULT_TS_MODEL_DIRECTORY)
    make_dir.assert_not_called()


@patch("sagemaker_inference.torchserve._generate_ts_config_properties")
@patch("sagemaker_inference.utils.write_file")
def test_create_torchserve_config_file(write_file, generate_ts_config_props):
    torchserve._create_torchserve_config_file()

    write_file.assert_called_once_with(
        torchserve.TS_CONFIG_FILE, generate_ts_config_props.return_value
    )


@patch("sagemaker_inference.utils.read_file", return_value=DEFAULT_CONFIGURATION)
@patch("sagemaker_inference.environment.Environment")
def test_generate_ts_config_properties(env, read_file):
    torchserve_timeout = "model_server_timeout"
    torchserve_workers = "model_server_workers"
    http_port = "http_port"

    env.return_value.model_server_timeout = torchserve_timeout
    env.return_value.model_server_workers = torchserve_workers
    env.return_value.inference_http_port = http_port

    ts_config_properties = torchserve._generate_ts_config_properties()

    inference_address = "inference_address=http://0.0.0.0:{}\n".format(http_port)
    server_timeout = "default_response_timeout={}\n".format(torchserve_timeout)
    workers = "default_workers_per_model={}\n".format(torchserve_workers)

    read_file.assert_called_once_with(torchserve.DEFAULT_TS_CONFIG_FILE)

    assert ts_config_properties.startswith(DEFAULT_CONFIGURATION)
    assert inference_address in ts_config_properties
    assert server_timeout in ts_config_properties
    assert workers in ts_config_properties


@patch("sagemaker_inference.utils.read_file", return_value=DEFAULT_CONFIGURATION)
@patch("sagemaker_inference.environment.Environment")
def test_generate_ts_config_properties_default_workers(env, read_file):
    env.return_value.torchserve_workers = None

    ts_config_properties = torchserve._generate_ts_config_properties()

    workers = "default_workers_per_model={}".format(None)

    read_file.assert_called_once_with(torchserve.DEFAULT_TS_CONFIG_FILE)

    assert ts_config_properties.startswith(DEFAULT_CONFIGURATION)
    assert workers not in ts_config_properties

