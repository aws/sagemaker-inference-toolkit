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
"""This module contains string constants that define inference toolkit
parameters."""
from __future__ import absolute_import

BASE_PATH_ENV = "SAGEMAKER_BASE_DIR"  # type: str
USER_PROGRAM_ENV = "SAGEMAKER_PROGRAM"  # type: str
LOG_LEVEL_ENV = "SAGEMAKER_CONTAINER_LOG_LEVEL"  # type: str
DEFAULT_INVOCATIONS_ACCEPT_ENV = "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT"  # type: str
MODEL_SERVER_WORKERS_ENV = "SAGEMAKER_MODEL_SERVER_WORKERS"  # type: str
MODEL_SERVER_TIMEOUT_ENV = "SAGEMAKER_MODEL_SERVER_TIMEOUT"  # type: str
STARTUP_TIMEOUT_ENV = "SAGEMAKER_STARTUP_TIMEOUT"  # type: str
BIND_TO_PORT_ENV = "SAGEMAKER_BIND_TO_PORT"  # type: str
SAFE_PORT_RANGE_ENV = "SAGEMAKER_SAFE_PORT_RANGE"  # type: str
MULTI_MODEL_ENV = "SAGEMAKER_MULTI_MODEL"  # type: str
