"""
ModelHandler defines an example model handler for load and inference requests for PyTorch CPU models
"""
import torch
import torchvision.models as models
import numpy as np


class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.pytorch_model = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        self.pytorch_model = models.resnet18(pretrained=True)

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it for inference

        img_list = []
        for idx, data in enumerate(request):
            # Read the bytearray of the image from the input
            img_arr = data.get("body")

            # Input image is in bytearray, convert it to PyTorch Tensor
            tensor = torch.ByteTensor(img_arr)
            if tensor is None:
                return None

            # convert into format (batch, RGB, width, height)
            tensor = tensor.reshape(224, 224)  # resize
            tensor = tensor.transpose(0, 1)
            tensor = tensor.transpose(0, 2)  # channel first
            tensor = tensor.unsqueeze(0)  # batchify
            img_list.append(tensor)

        return img_list

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        self.pytorch_model.eval()
        with torch.no_grad():
            output = self.pytorch_model(model_input)
        return output

    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        prob = np.squeeze(inference_output)
        a = np.argsort(prob)[::-1]
        return [["probability=%f, class=%s" % (prob[i], self.labels[i]) for i in a[0:5]]]

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """

        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)
