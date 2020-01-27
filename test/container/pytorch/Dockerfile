# Get MMS Base Python 3.6 CPU image on Ubuntu 16.04 base
FROM awsdeeplearningteam/mxnet-model-server:base-cpu-py3.6

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

COPY pytorch/mme_handler_service.py /mme_handler_service.py

# Setup Environment for MME
ENV SAGEMAKER_MULTI_MODEL=true
ENV SAGEMAKER_BIND_TO_PORT=${SAGEMAKER_BIND_TO_PORT:-8080}

# Update MMS version
RUN pip3 install mxnet-model-server==1.0.8

# Install PyTorch (for handler_service)
RUN pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /

COPY sagemaker_inference.tar.gz /sagemaker_inference.tar.gz

RUN pip3 install --no-cache-dir /sagemaker_inference.tar.gz \
 && rm /sagemaker_inference.tar.gz

COPY pytorch/mms-entrypoint.py /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/
COPY pytorch/mme_handler_service.py /home/model-server/mme_handler_service.py
ENV SAGEMAKER_HANDLER="/home/model-server/mme_handler_service.py:handle"

EXPOSE 8080

ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
CMD ["serve"]
