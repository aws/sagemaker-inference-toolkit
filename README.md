![SageMaker](https://github.com/aws/sagemaker-inference-toolkit/raw/master/branding/icon/sagemaker-banner.png)

# SageMaker Inference Toolkit

[![Latest Version](https://img.shields.io/pypi/v/sagemaker-inference.svg)](https://pypi.python.org/pypi/sagemaker-inference) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker-inference.svg)](https://pypi.python.org/pypi/sagemaker-inference) [![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)

Serve machine learning models within a Docker container using Amazon
SageMaker.


## :books: Background

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service for data science and machine learning (ML) workflows.
You can use Amazon SageMaker to simplify the process of building, training, and deploying ML models.

Once you have a trained model, you can include it in a [Docker container](https://www.docker.com/resources/what-container) that runs your inference code.
A container provides an effectively isolated environment, ensuring a consistent runtime regardless of where the container is deployed.
Containerizing your model and code enables fast and reliable deployment of your model.

The **SageMaker Inference Toolkit** implements a model serving stack and can be easily added to any Docker container, making it [deployable to SageMaker](https://aws.amazon.com/sagemaker/deploy/).
This library's serving stack is built on [Multi Model Server](https://github.com/awslabs/mxnet-model-server), and it can serve your own models or those you trained on SageMaker using [machine learning frameworks with native SageMaker support](https://docs.aws.amazon.com/sagemaker/latest/dg/frameworks.html).
If you use a [prebuilt SageMaker Docker image for inference](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html), this library may already be included.

For more information, see the Amazon SageMaker Developer Guide sections on [building your own container with Multi Model Server](https://docs.aws.amazon.com/sagemaker/latest/dg/build-multi-model-build-container.html) and [using your own models](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html).

## :hammer_and_wrench: Installation

To install this library in your Docker image, add the following line to your [Dockerfile](https://docs.docker.com/engine/reference/builder/):

``` dockerfile
RUN pip3 install multi-model-server sagemaker-inference-toolkit
```

[Here is an example](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/multi_model_bring_your_own/container/Dockerfile) of a Dockerfile that installs SageMaker Inference Toolkit.

## :computer: Usage

### Implementation Steps

To use the SageMaker Inference Toolkit, you need to do the following:

1.  Implement an inference handler, which is responsible for loading the model and providing input, predict, and output functions.
    ([Here is an example](https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py) of an inference handler.)

    ``` python
    from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors

    class DefaultPytorchInferenceHandler(default_inference_handler.DefaultInferenceHandler):

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
    ```

2.  Implement a handler service that is executed by the model server.
    ([Here is an example](https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/handler_service.py) of a handler service.)
    For more information on how to define your `HANDLER_SERVICE` file, see [the MMS custom service documentation](https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md).

    ``` python
    from sagemaker_inference.default_handler_service import DefaultHandlerService
    from sagemaker_inference.transformer import Transformer
    from sagemaker_pytorch_serving_container.default_inference_handler import DefaultPytorchInferenceHandler


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
    ```

3.  Implement a serving entrypoint, which starts the model server.
    ([Here is an example](https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/serving.py) of a serving entrypoint.)

    ``` python
    from sagemaker_inference import model_server

    model_server.start_model_server(handler_service=HANDLER_SERVICE)
    ```

4.  Define the location of the entrypoint in your Dockerfile.

    ``` dockerfile
    ENTRYPOINT ["python", "/usr/local/bin/entrypoint.py"]
    ```

### Complete Example

[Here is a complete example](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/multi_model_bring_your_own) demonstrating usage of the SageMaker Inference Toolkit in your own container for deployment to a multi-model endpoint.

## :scroll: License

This library is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).
For more details, please take a look at the [LICENSE](https://github.com/aws-samples/sagemaker-inference-toolkit/blob/master/LICENSE) file.

## :handshake: Contributing

Contributions are welcome!
Please read our [contributing guidelines](https://github.com/aws/sagemaker-inference-toolkit/blob/master/CONTRIBUTING.md)
if you'd like to open an issue or submit a pull request.
