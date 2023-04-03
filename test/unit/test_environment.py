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

from mock import patch
import pytest

from sagemaker_inference import environment, parameters


@patch.dict(
    os.environ,
    {
        parameters.USER_PROGRAM_ENV: "main.py",
        parameters.MODEL_SERVER_TIMEOUT_ENV: "20",
        parameters.MODEL_SERVER_WORKERS_ENV: "8",
        parameters.STARTUP_TIMEOUT_ENV: "50",
        parameters.DEFAULT_INVOCATIONS_ACCEPT_ENV: "text/html",
        parameters.BIND_TO_PORT_ENV: "1738",
        parameters.SAFE_PORT_RANGE_ENV: "1111-2222",
        parameters.MODEL_SERVER_VMARGS: "-XX:-UseContainerSupport",
        parameters.MAX_REQUEST_SIZE: "10",
    },
    clear=True,
)
def test_env():
    env = environment.Environment()

    assert environment.base_dir.endswith("/opt/ml")
    assert environment.model_dir.endswith("/opt/ml/model")
    assert environment.code_dir.endswith("opt/ml/model/code")
    assert env.module_name == "main"
    assert env.model_server_timeout == 20
    assert env.startup_timeout == 50
    assert env.model_server_workers == "8"
    assert env.default_accept == "text/html"
    assert env.inference_http_port == "1738"
    assert env.management_http_port == "1738"
    assert env.safe_port_range == "1111-2222"
    assert "-XX:-UseContainerSupport" in env.vmargs
    assert env.max_request_size == 10 * 1024 * 1024


@pytest.mark.parametrize("sagemaker_program", ["program.py", "program"])
@patch.dict(os.environ, {}, clear=True)
def test_env_module_name(sagemaker_program):
    os.environ[parameters.USER_PROGRAM_ENV] = sagemaker_program
    module_name = environment.Environment().module_name

    del os.environ[parameters.USER_PROGRAM_ENV]

    assert module_name == "program"
