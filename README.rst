===========================
SageMaker Inference Toolkit
===========================

.. image:: https://img.shields.io/badge/code_style-black-000000.svg
   :target: https://github.com/python/black
   :alt: Code style: black

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

Using SageMaker Inference Toolkit
---------------------------------

SageMaker Inference Toolkit serving stack leverages `Model Server for Apache MXNet (MMS) <https://github.com/awslabs/mxnet-model-server>`_ to server deep learning models trained using any ML/DL framework in SageMaker. Any SageMaker container can use the SageMaker Inference Toolkit to implement their serving stack. To use the Inference Toolkit, customers need to implement the following the components:

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


License
-------

This library is licensed under the Apache 2.0 License.
It is copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
The license is available at: http://aws.amazon.com/apache2.0/
