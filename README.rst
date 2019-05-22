===========================
SageMaker Inference Toolkit
===========================

SageMaker Inference Toolkit is a library used for enabling serving within the SageMaker prebuilt deep learning framework containers.

This library is the serving subset of the `SageMaker Containers library <https://github.com/aws/sagemaker-containers>`__.

Currently, this library is used by the following containers:

- `SageMaker MXNet Serving Container <https://github.com/aws/sagemaker-mxnet-serving-container>`__

-----------------
Table of Contents
-----------------
.. contents::
    :local:

Getting Started
---------------

The main purpose of this library is to start up a model server within a container to enable serving on SageMaker.

This library assumes the following `SageMaker inference requirements <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html>`__ are met.

The following code block shows how to start the model server.

.. code:: python

    from sagemaker_inference import model_server

    model_server.start_model_server(handler_service=HANDLER_SERVICE)

The ``HANDLER_SERVICE`` is a string literal that points to the Python path of a Python file that will be executed by the
model server for incoming invocation requests. This Python script is responsible for handling incoming data and passing it on to the engine for inference.
The Python file should define a ``handle`` method that acts as an entry point for execution, this function will be invoked by the model server on a inference request.

For more information on how to define your ``HANDLER_SERVICE`` file, see `Custom Service <https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md>`__.

Running tests
-------------

To run all tests:

::

    tox

License
-------

This library is licensed under the Apache 2.0 License.
It is copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
The license is available at: http://aws.amazon.com/apache2.0/
