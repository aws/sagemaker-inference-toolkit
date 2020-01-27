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

import shlex
import subprocess
from subprocess import CalledProcessError
import sys
from time import sleep

from sagemaker_inference import default_handler_service
from sagemaker_inference import model_server

HANDLER_SERVICE = default_handler_service.__name__


def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError)


def _start_model_server():
    sleep(5)
    model_server.start_model_server(handler_service=HANDLER_SERVICE)


if sys.argv[1] == "serve":
    _start_model_server()
else:
    subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))

# prevent docker exit
subprocess.call(["tail", "-f", "/dev/null"])
