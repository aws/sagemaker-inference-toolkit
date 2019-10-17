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
import os
import signal
import subprocess
import types

from mock import Mock, patch
import pytest

from sagemaker_inference import environment, model_server
from sagemaker_inference.model_server import MMS_NAMESPACE
from sagemaker_inference.model_server import REQUIREMENTS_PATH

PYTHON_PATH = 'python_path'
DEFAULT_CONFIGURATION = 'default_configuration'


@patch('subprocess.call')
@patch('subprocess.Popen')
@patch('sagemaker_inference.model_server._retrieve_mms_server_process')
@patch('sagemaker_inference.model_server._add_sigterm_handler')
@patch('sagemaker_inference.model_server._install_requirements')
@patch('os.path.exists', return_value=True)
@patch('sagemaker_inference.model_server._create_model_server_config_file')
@patch('sagemaker_inference.model_server._adapt_to_mms_format')
def test_start_model_server_default_service_handler(adapt, create_config, exists, install_requirements, sigterm,
                                                    retrieve, subprocess_popen, subprocess_call):
    model_server.start_model_server()

    adapt.assert_called_once_with(model_server.DEFAULT_HANDLER_SERVICE)
    create_config.assert_called_once()
    exists.assert_called_once_with(REQUIREMENTS_PATH)
    install_requirements.assert_called_once()

    mxnet_model_server_cmd = ['mxnet-model-server',
                              '--start',
                              '--model-store', model_server.DEFAULT_MMS_MODEL_DIRECTORY,
                              '--mms-config', model_server.MMS_CONFIG_FILE,
                              '--log-config', model_server.DEFAULT_MMS_LOG_FILE,
                              ]

    subprocess_popen.assert_called_once_with(mxnet_model_server_cmd)
    sigterm.assert_called_once_with(retrieve.return_value)


@patch('subprocess.call')
@patch('subprocess.Popen')
@patch('sagemaker_inference.model_server._retrieve_mms_server_process')
@patch('sagemaker_inference.model_server._add_sigterm_handler')
@patch('sagemaker_inference.model_server._create_model_server_config_file')
@patch('sagemaker_inference.model_server._adapt_to_mms_format')
def test_start_model_server_custom_handler_service(adapt, create_config, sigterm, retrieve, subprocess_popen,
                                                   subprocess_call):
    handler_service = Mock()

    model_server.start_model_server(handler_service)

    adapt.assert_called_once_with(handler_service)


@patch('sagemaker_inference.model_server._set_python_path')
@patch('subprocess.check_call')
@patch('os.makedirs')
@patch('os.path.exists', return_value=False)
def test_adapt_to_mms_format(path_exists, make_dir, subprocess_check_call, set_python_path):
    handler_service = Mock()

    model_server._adapt_to_mms_format(handler_service)

    path_exists.assert_called_once_with(model_server.DEFAULT_MMS_MODEL_DIRECTORY)
    make_dir.assert_called_once_with(model_server.DEFAULT_MMS_MODEL_DIRECTORY)

    model_archiver_cmd = ['model-archiver',
                          '--model-name', model_server.DEFAULT_MMS_MODEL_NAME,
                          '--handler', handler_service,
                          '--model-path', environment.model_dir,
                          '--export-path', model_server.DEFAULT_MMS_MODEL_DIRECTORY,
                          '--archive-format', 'no-archive',
                          ]

    subprocess_check_call.assert_called_once_with(model_archiver_cmd)
    set_python_path.assert_called_once()


@patch('sagemaker_inference.model_server._set_python_path')
@patch('subprocess.check_call')
@patch('os.makedirs')
@patch('os.path.exists', return_value=True)
def test_adapt_to_mms_format_existing_path(path_exists, make_dir, subprocess_check_call, set_python_path):
    handler_service = Mock()

    model_server._adapt_to_mms_format(handler_service)

    path_exists.assert_called_once_with(model_server.DEFAULT_MMS_MODEL_DIRECTORY)
    make_dir.assert_not_called()


@patch.dict(os.environ, {model_server.PYTHON_PATH_ENV: PYTHON_PATH}, clear=True)
def test_set_existing_python_path():
    model_server._set_python_path()

    code_dir_path = '{}:'.format(environment.code_dir)

    assert os.environ[model_server.PYTHON_PATH_ENV] == code_dir_path + PYTHON_PATH


@patch.dict(os.environ, {}, clear=True)
def test_new_python_path():
    model_server._set_python_path()

    code_dir_path = '{}:'.format(environment.code_dir)

    assert os.environ[model_server.PYTHON_PATH_ENV] == code_dir_path


@patch('sagemaker_inference.model_server._generate_mms_config_properties')
@patch('sagemaker_inference.utils.write_file')
def test_create_model_server_config_file(write_file, generate_mms_config_props):
    model_server._create_model_server_config_file()

    write_file.assert_called_once_with(model_server.MMS_CONFIG_FILE, generate_mms_config_props.return_value)


@patch('sagemaker_inference.utils.read_file', return_value=DEFAULT_CONFIGURATION)
@patch('sagemaker_inference.environment.Environment')
def test_generate_mms_config_properties(env, read_file):
    model_server_timeout = 'model_server_timeout'
    model_server_workers = 'model_server_workers'
    http_port = 'http_port'

    env.return_value.model_server_timeout = model_server_timeout
    env.return_value.model_server_workers = model_server_workers
    env.return_value.http_port = http_port

    mms_config_properties = model_server._generate_mms_config_properties()

    inference_address = 'inference_address=http://0.0.0.0:{}\n'.format(http_port)
    server_timeout = 'default_response_timeout={}\n'.format(model_server_timeout)
    workers = 'default_workers_per_model={}\n'.format(model_server_workers)

    read_file.assert_called_once_with(model_server.DEFAULT_MMS_CONFIG_FILE)

    assert mms_config_properties.startswith(DEFAULT_CONFIGURATION)
    assert inference_address in mms_config_properties
    assert server_timeout in mms_config_properties
    assert workers in mms_config_properties


@patch('sagemaker_inference.utils.read_file', return_value=DEFAULT_CONFIGURATION)
@patch('sagemaker_inference.environment.Environment')
def test_generate_mms_config_properties_default_workers(env, read_file):
    env.return_value.model_server_workers = None

    mms_config_properties = model_server._generate_mms_config_properties()

    workers = 'default_workers_per_model={}'.format(None)

    read_file.assert_called_once_with(model_server.DEFAULT_MMS_CONFIG_FILE)

    assert mms_config_properties.startswith(DEFAULT_CONFIGURATION)
    assert workers not in mms_config_properties


@patch('signal.signal')
def test_add_sigterm_handler(signal_call):
    mms = Mock()

    model_server._add_sigterm_handler(mms)

    mock_calls = signal_call.mock_calls
    first_argument = mock_calls[0][1][0]
    second_argument = mock_calls[0][1][1]

    assert len(mock_calls) == 1
    assert first_argument == signal.SIGTERM
    assert isinstance(second_argument, types.FunctionType)


@patch('subprocess.check_call')
def test_install_requirements(check_call):
    model_server._install_requirements()


@patch('subprocess.check_call', side_effect=subprocess.CalledProcessError(0, "cmd"))
def test_install_requirements_installation_failed(check_call):
    with pytest.raises(ValueError) as e:
        model_server._install_requirements()

    assert 'failed to install required packages' in str(e.value)


@patch('retrying.Retrying.should_reject', return_value=False)
@patch('psutil.process_iter')
def test_retrieve_mms_server_process(process_iter, retry):
    server = Mock()
    server.cmdline.return_value = MMS_NAMESPACE

    processes = list()
    processes.append(server)

    process_iter.return_value = processes

    process = model_server._retrieve_mms_server_process()

    assert process == server


@patch('retrying.Retrying.should_reject', return_value=False)
@patch('psutil.process_iter', return_value=list())
def test_retrieve_mms_server_process_no_server(process_iter, retry):
    with pytest.raises(Exception) as e:
        model_server._retrieve_mms_server_process()

    assert 'mms model server was unsuccessfully started' in str(e.value)


@patch('retrying.Retrying.should_reject', return_value=False)
@patch('psutil.process_iter')
def test_retrieve_mms_server_process_too_many_servers(process_iter, retry):
    server = Mock()
    second_server = Mock()
    server.cmdline.return_value = MMS_NAMESPACE
    second_server.cmdline.return_value = MMS_NAMESPACE

    processes = list()
    processes.append(server)
    processes.append(second_server)

    process_iter.return_value = processes

    with pytest.raises(Exception) as e:
        model_server._retrieve_mms_server_process()

    assert 'multiple mms model servers are not supported' in str(e.value)
