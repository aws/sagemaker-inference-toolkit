# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import os
import signal
import subprocess

import pkg_resources

import sagemaker_inference
from sagemaker_inference import default_handler_service, environment, logging, utils

logger = logging.get_logger()

MMS_CONFIG_FILE = os.path.join('/etc', 'sagemaker-mms.properties')
DEFAULT_HANDLER_SERVICE = default_handler_service.__name__
DEFAULT_MMS_CONFIG_FILE = pkg_resources.resource_filename(sagemaker_inference.__name__, '/etc/default-mms.properties')
DEFAULT_MMS_LOG_FILE = pkg_resources.resource_filename(sagemaker_inference.__name__, '/etc/log4j.properties')
DEFAULT_MMS_MODEL_DIRECTORY = os.path.join(os.getcwd(), '.sagemaker/mms/models')
DEFAULT_MMS_MODEL_NAME = 'model'

PYTHON_PATH_ENV = 'PYTHONPATH'


def start_model_server(handler_service=DEFAULT_HANDLER_SERVICE):
    """Configure and start the model server.

    Args:
        handler_service (str): python path pointing to a module that defines a class with the following:
            - A ``handle`` method, which is invoked for all incoming inference requests to the model server.
            - A ``initialize`` method, which is invoked at model server start up for loading the model.
            Defaults to ``sagemaker_inference.default_handler_service``.

    """
    _adapt_to_mms_format(handler_service)
    _create_model_server_config_file()

    mxnet_model_server_cmd = ['mxnet-model-server',
                              '--start',
                              '--model-store', DEFAULT_MMS_MODEL_DIRECTORY,
                              '--mms-config', MMS_CONFIG_FILE,
                              '--log-config', DEFAULT_MMS_LOG_FILE,
                              ]

    logger.info(mxnet_model_server_cmd)
    mms_process = subprocess.Popen(mxnet_model_server_cmd)

    _add_sigterm_handler(mms_process)

    subprocess.call(['tail',
                     '-f',
                     '/dev/null'])


def _adapt_to_mms_format(handler_service):
    if not os.path.exists(DEFAULT_MMS_MODEL_DIRECTORY):
        os.makedirs(DEFAULT_MMS_MODEL_DIRECTORY)

    model_archiver_cmd = ['model-archiver',
                          '--model-name', DEFAULT_MMS_MODEL_NAME,
                          '--handler', handler_service,
                          '--model-path', environment.model_dir,
                          '--export-path', DEFAULT_MMS_MODEL_DIRECTORY,
                          '--archive-format', 'no-archive',
                          ]

    logger.info(model_archiver_cmd)
    subprocess.check_call(model_archiver_cmd)

    _set_python_path()


def _set_python_path():
    # MMS handles code execution by appending the export path, provided
    # to the model archiver, to the PYTHONPATH env var.
    # The code_dir has to be added to the PYTHONPATH otherwise the
    # user provided module can not be imported properly.
    code_dir_path = '{}:'.format(environment.code_dir)

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
        'default_response_timeout': env.model_server_timeout,
        'default_workers_per_model': env.model_server_workers,
        'inference_address': 'http://0.0.0.0:{}'.format(env.http_port)
    }

    custom_configuration = str()

    for key in user_defined_configuration:
        value = user_defined_configuration.get(key)
        if value:
            custom_configuration += '{}={}\n'.format(key, value)

    default_configuration = utils.read_file(DEFAULT_MMS_CONFIG_FILE)

    return default_configuration + custom_configuration


def _add_sigterm_handler(mms_process):
    def _terminate(signo, frame):
        try:
            os.kill(mms_process.pid, signal.SIGTERM)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _terminate)
