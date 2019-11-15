# Get MMS Base Python 3.6 CPU image on Ubuntu 16.04 base
FROM awsdeeplearningteam/mxnet-model-server:base-cpu-py3.6

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

COPY mme_handler_service.py /mme_handler_service.py

# Setup Environment for MME
ENV SAGEMAKER_MULTI_MODEL=true
ENV SAGEMAKER_BIND_TO_PORT=${SAGEMAKER_BIND_TO_PORT:-8080}

# Update MMS version
RUN pip3 install mxnet-model-server==1.0.8

# Install Mxnet (for handler_service)
RUN pip3 install mxnet

WORKDIR /

COPY sagemaker_inference.tar.gz /sagemaker_inference.tar.gz

RUN pip3 install --no-cache-dir /sagemaker_inference.tar.gz \
 && rm /sagemaker_inference.tar.gz

# download models to model_store
RUN mkdir resnet_152 \
 && cd resnet_152 \
 && wget -O resnet-152-0000.params http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-0000.params \
 && wget -O resnet-152-symbol.json http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-symbol.json \
 && wget -O synset.txt http://data.mxnet.io/models/imagenet/synset.txt \
 && echo '[{"shape": [1, 3, 224, 224], "name": "data"}]' > resnet-152-shapes.json \
 && cd ..

RUN mkdir resnet_18 \
 && cd resnet_18 \
 && wget -O resnet-18-0000.params http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params \
 && wget -O resnet-18-symbol.json http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json \
 && wget -O synset.txt http://data.mxnet.io/models/imagenet/synset.txt \
 && echo '[{"shape": [1, 3, 224, 224], "name": "data"}]' > resnet-18-shapes.json \
 && cd ..

COPY dummy/mms-entrypoint.py /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/
COPY dummy/mme_handler_service.py /home/model-server/mme_handler_service.py
ENV SAGEMAKER_HANDLER="/home/model-server/mme_handler_service.py:handle"

EXPOSE 8080

ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
CMD ["serve"]
