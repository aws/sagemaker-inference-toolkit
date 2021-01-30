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
from __future__ import absolute_import

from glob import glob
import os
import sys

import setuptools


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


def read_version():
    return read("VERSION").strip()


packages = setuptools.find_packages(where="src", exclude=("test",))

required_packages = ["numpy", "six", "psutil", "retrying==1.3.3", "scipy"]

# enum is introduced in Python 3.4. Installing enum back port
if sys.version_info < (3, 4):
    required_packages.append("enum34 >= 1.1.6")

PKG_NAME = "sagemaker_inference"

setuptools.setup(
    name=PKG_NAME,
    version=read_version(),
    description="Open source toolkit for helping create serving containers to run on Amazon SageMaker.",
    packages=packages,
    package_dir={PKG_NAME: "src/sagemaker_inference"},
    package_data={PKG_NAME: ["etc/*"]},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Amazon Web Services",
    url="https://github.com/aws/sagemaker-inference-toolkit/",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=required_packages,
    extras_require={
        "test": ["tox", "flake8", "pytest", "pytest-xdist", "pytest-cov", "mock", "requests"]
    },
)
