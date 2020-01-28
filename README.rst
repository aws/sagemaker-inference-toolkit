.. image:: https://github.com/aws/sagemaker-inference-toolkit/raw/master/branding/icon/sagemaker-banner.png
    :height: 100px
    :alt: SageMaker

===========================
SageMaker Inference Toolkit
===========================

.. image:: https://img.shields.io/pypi/v/sagemaker-inference.svg
   :target: https://pypi.python.org/pypi/sagemaker-inference
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/sagemaker-inference.svg
   :target: https://pypi.python.org/pypi/sagemaker-inference
   :alt: Supported Python Versions

.. image:: https://img.shields.io/badge/code_style-black-000000.svg
   :target: https://github.com/python/black
   :alt: Code style: black

**SageMaker Inference Toolkit** is an open-source library for serving machine learning models within a Docker container.

This library's serving stack is built on `Multi Model Server (MMS) <https://github.com/awslabs/mxnet-model-server>`_ to serve machine learning models trained using `any machine learning framework in SageMaker <https://docs.aws.amazon.com/sagemaker/latest/dg/frameworks.html>`__.
SageMaker-compatible Docker containers can use the SageMaker Inference Toolkit to implement their serving stack. (It is used in some of the `prebuilt SageMaker Docker images for inference <https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html>`__.)

-----------------
Table of Contents
-----------------
.. contents::
    :local:

Getting Started
---------------

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

Using SageMaker Inference Toolkit
---------------------------------

To use the Inference Toolkit, customers need to implement the following the components:

- An inference handler responsible to load the model, and provide default input, predict, and output functions:

.. code:: python

    from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors

    class DefaultPytorchInferenceHandler(default_inference_handler.DefaultInferenceHandler):
        VALID_CONTENT_TYPES = (content_types.JSON, content_types.NPY)

        def default_model_fn(self, model_dir):
            """Loads a model. For PyTorch, a default function to load a model cannot be provided.
            Users should provide customized model_fn() in script.
            Args:
                model_dir: a directory where model is saved.
            Returns: A PyTorch model.
            """
            raise NotImplementedError(textwrap.dedent("""
            Please provide a model_fn implementation.
            See documentation for model_fn at https://github.com/aws/sagemaker-python-sdk
            """))

        def default_input_fn(self, input_data, content_type):
            """A default input_fn that can handle JSON, CSV and NPZ formats.
            Args:
                input_data: the request payload serialized in the content_type format
                content_type: the request content_type
            Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
            """
            return decoder.decode(input_data, content_type)

        def default_predict_fn(self, data, model):
            """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
            Runs prediction on GPU if cuda is available.
            Args:
                data: input data (torch.Tensor) for prediction deserialized by input_fn
                model: PyTorch model loaded in memory by model_fn
            Returns: a prediction
            """
            return model(input_data)

        def default_output_fn(self, prediction, accept):
            """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.
            Args:
                prediction: a prediction result from predict_fn
                accept: type which the output data needs to be serialized
            Returns: output data serialized
            """
            return encoder.encode(prediction, accept)

- A handler service that is executed by the model server:

.. code:: python

    from sagemaker_inference.default_handler_service import DefaultHandlerService
    from sagemaker_inference.transformer import Transformer
    from sagemaker_pytorch_serving_container.default_inference_handler import \
        DefaultPytorchInferenceHandler


    class HandlerService(DefaultHandlerService):
        """Handler service that is executed by the model server.
        Determines specific default inference handlers to use based on model being used.
        This class extends ``DefaultHandlerService``, which define the following:
            - The ``handle`` method is invoked for all incoming inference requests to the model server.
            - The ``initialize`` method is invoked at model server start up.
        Based on: https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md
        """
        def __init__(self):
            transformer = Transformer(default_inference_handler=DefaultPytorchInferenceHandler())
            super(HandlerService, self).__init__(transformer=transformer)


- A serving entrypoint responsible to start MMS:

.. code:: python

    from sagemaker_inference import model_server
    
    def main():
        model_server.start_model_server(handler_service=HANDLER_SERVICE)


Complete example `https://github.com/aws/sagemaker-pytorch-serving-container/pull/4/files`

