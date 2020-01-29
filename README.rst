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
   :alt: Code Style: Black

Serve machine learning models within a Docker container using Amazon SageMaker.

-----------------
Table of Contents
-----------------
.. contents::
    :local:

Background
----------

`Amazon SageMaker <https://aws.amazon.com/sagemaker/>`__ is a fully managed service that provides every developer and data scientist with the ability to build, train, and `deploy <https://aws.amazon.com/sagemaker/deploy/>`__ machine learning models quickly.
SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop and deploy high quality models.

Once you've trained a machine learning model, you can deploy it to a `Docker container <https://www.docker.com/resources/what-container>`__, where you can run your own inference code.
The code that runs in a container is effectively isolated from its surroundings, ensuring a consistent runtime, regardless of where the container is deployed.
By using a container, you can deploy machine learning models quickly and reliably at any scale.
Additionally, you will be able to use your container to create a `Multi-Model Endpoint <https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html>`__, which enables you to deploy multiple models on a single endpoint and serve them using a single container.

The **SageMaker Inference Toolkit** library allows you to implement the model serving stack in your Docker container, making it compatible with SageMaker Multi Model Endpoints.
This library's serving stack is built on `Multi Model Server (MMS) <https://github.com/awslabs/mxnet-model-server>`_, and it can serve your own models or those you trained on SageMaker using `any machine learning framework <https://docs.aws.amazon.com/sagemaker/latest/dg/frameworks.html>`__.
(It is used in some of the `prebuilt SageMaker Docker images for inference <https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html>`__.)

For more background information, please consult the Amazon SageMaker Developer Guide sections on `building your own container with Multi Model Server <https://docs.aws.amazon.com/sagemaker/latest/dg/build-multi-model-build-container.html>`__ and `using your own models <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html>`__.

Installation
------------

To install this library in your Docker container, add the following line to your `Dockerfile <https://docs.docker.com/engine/reference/builder/>`__:

.. code:: dockerfile

    RUN pip3 install multi-model-server sagemaker-inference-toolkit

`Here is an example <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/multi_model_bring_your_own/container/Dockerfile>`__ of a Dockerfile that installs SageMaker Inference Toolkit.

Usage
-----

The `Amazon SageMaker Developer Guide <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html>`__ explains how SageMaker interacts with a Docker container that runs your own inference code for hosting services. Use this information to write inference code and create a Docker image.

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

    model_server.start_model_server(handler_service=HANDLER_SERVICE)

The ``HANDLER_SERVICE`` is a string literal that points to the Python path of a Python file that will be executed by the
model server for incoming invocation requests. This Python script is responsible for handling incoming data and passing it on to the engine for inference.
The Python file should define a ``handle`` method that acts as an entry point for execution, this function will be invoked by the model server on a inference request.

For more information on how to define your ``HANDLER_SERVICE`` file, see `Custom Service <https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md>`__.


`Here is a complete example <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/multi_model_bring_your_own>`__ of using the SageMaker Inference Toolkit in your own container for deployment to a Multi-Model Endpoint.

License
-------
This library is licensed under the `Apache 2.0 License <http://aws.amazon.com/apache2.0/>`__.
For more details, please take a look at the `LICENSE <https://github.com/aws-samples/sagemaker-inference-toolkit/blob/master/LICENSE>`__ file.

Contributing
------------

Contributions are welcome! Please read our `contributing guidelines <https://github.com/aws/sagemaker-inference-toolkit/blob/master/CONTRIBUTING.md>`__ if you'd like to open an issue or submit a pull request.