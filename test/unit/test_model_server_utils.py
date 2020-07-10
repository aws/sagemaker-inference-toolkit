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

@patch.dict(os.environ, {torchserve.PYTHON_PATH_ENV: PYTHON_PATH}, clear=True)
def test_set_existing_python_path():
    torchserve.set_python_path()

    code_dir_path = "{}:".format(environment.code_dir)

    assert os.environ[torchserve.PYTHON_PATH_ENV] == code_dir_path + PYTHON_PATH


@patch.dict(os.environ, {}, clear=True)
def test_new_python_path():
    torchserve.set_python_path()

    code_dir_path = "{}:".format(environment.code_dir)

    assert os.environ[torchserve.PYTHON_PATH_ENV] == code_dir_path


@patch("signal.signal")
def testadd_sigterm_handler(signal_call):
    ts = Mock()

    torchserve.add_sigterm_handler(ts)

    mock_calls = signal_call.mock_calls
    first_argument = mock_calls[0][1][0]
    second_argument = mock_calls[0][1][1]

    assert len(mock_calls) == 1
    assert first_argument == signal.SIGTERM
    assert isinstance(second_argument, types.FunctionType)


@patch("subprocess.check_call")
def testinstall_requirements(check_call):
    torchserve.install_requirements()


@patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(0, "cmd"))
def testinstall_requirements_installation_failed(check_call):
    with pytest.raises(ValueError) as e:
        torchserve.install_requirements()

    assert "failed to install required packages" in str(e.value)


@patch("retrying.Retrying.should_reject", return_value=False)
@patch("psutil.process_iter")
def test_retrieve_model_server_process(process_iter, retry):
    server = Mock()
    server.cmdline.return_value = TS_NAMESPACE

    processes = list()
    processes.append(server)

    process_iter.return_value = processes

    process = model_server_utils.retrieve_model_server_process(TS_NAMESPACE)

    assert process == server


@patch("retrying.Retrying.should_reject", return_value=False)
@patch("psutil.process_iter", return_value=list())
def test_retrieve_model_server_process_no_server(process_iter, retry):
    with pytest.raises(Exception) as e:
        model_server_utils.retrieve_model_server_process(TS_NAMESPACE)

    assert "model server was unsuccessfully started" in str(e.value)


@patch("retrying.Retrying.should_reject", return_value=False)
@patch("psutil.process_iter")
def test_retrieve_model_server_process_too_many_servers(process_iter, retry):
    server = Mock()
    second_server = Mock()
    server.cmdline.return_value = TS_NAMESPACE
    second_server.cmdline.return_value = TS_NAMESPACE

    processes = list()
    processes.append(server)
    processes.append(second_server)

    process_iter.return_value = processes

    with pytest.raises(Exception) as e:
        torchserve.retrieve_model_server_process(TS_NAMESPACE)

    assert "multiple model servers are not supported" in str(e.value)
