![SageMaker](https://github.com/aws/sagemaker-inference-toolkit/raw/master/branding/icon/sagemaker-banner.png)

# SageMaker Inference Toolkit

[![Latest Version](https://img.shields.io/pypi/v/sagemaker-inference.svg)](https://pypi.python.org/pypi/sagemaker-inference) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker-inference.svg)](https://pypi.python.org/pypi/sagemaker-inference) [![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)

Amazon SageMaker kullanarak Docker konteynerinde makine öğrenimi modellerini sunar.

## :books: Arka Plan

[Amazon SageMaker](https://aws.amazon.com/sagemaker/), veri bilimi ve makine öğrenimi (ML) iş akışları için tamamen yönetilen bir hizmettir.
Amazon SageMaker'ı kullanarak ML modelleri oluşturma, eğitme ve dağıtma sürecini basitleştirebilirsiniz.

Eğitilmiş bir modele sahip olduktan sonra, çıkarım kodunu çalıştıran bir [Docker konteynerine](https://www.docker.com/resources/what-container) dahil edebilirsiniz.
Bir konteyner, etkili bir şekilde izole edilmiş bir ortam sağlar ve konteynerin nerede dağıtıldığına bağlı olarak tutarlı bir çalışma zamanı sağlar.
Modelinizi ve kodunuzu konteynerleştirmek, modelinizi hızlı ve güvenilir bir şekilde dağıtmanızı sağlar.

**SageMaker Inference Toolkit** makine öğrenimi modeli sunan bir yığını uygular ve herhangi bir Docker konteynerine kolayca ekleyebilir, böylece [SageMaker'a](https://aws.amazon.com/sagemaker/deploy/) dağıtılabilir hale getirir.
Bu kitaplığın sunum yığını, [Multi Model Server](https://github.com/awslabs/multi-model-server) üzerine inşa edilmiştir ve kendi modellerinizi veya [SageMaker'da eğittiğiniz makine öğrenimi çerçevelerini](https://docs.aws.amazon.com/sagemaker/latest/dg/frameworks.html) sunabilir.
[Önceden oluşturulmuş SageMaker Docker görüntüsü](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html) kullanıyorsanız, bu kitaplık zaten dahil edilmiş olabilir.

Daha fazla bilgi için, Amazon SageMaker Geliştirici Kılavuzu'ndaki [Multi Model Server ile kendi konteynerinizi oluşturma](https://docs.aws.amazon.com/sagemaker/latest/dg/build-multi-model-build-container.html) ve [kendi modellerinizi kullanma](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html) bölümlerine bakın.

## :hammer_and_wrench: Kurulum

Bu kitaplığı Docker görüntünüze kurmak için, aşağıdaki satırı [Dockerfile](https

://docs.docker.com/engine/reference/builder/) dosyanıza ekleyin:

``` dockerfile
RUN pip3 install multi-model-server sagemaker-inference
```

SageMaker Inference Toolkit'i kurulu olan bir Dockerfile örneği için [buraya](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/multi_model_bring_your_own/container/Dockerfile) göz atabilirsiniz.

## :computer: Kullanım

### Uygulama Adımları

SageMaker Inference Toolkit'i kullanmak için aşağıdaki adımları uygulamanız gerekmektedir:

1.  Model yükleme ve giriş, tahmin ve çıktı işlevlerini sağlama sorumluluğunu üstlenen bir çıkarım işleyici (inference handler) uygulayın.
    ([İşte bir örnek](https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/default_pytorch_inference_handler.py) bir çıkarım işleyici örneği.)

    ``` python
    from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors

    class DefaultPytorchInferenceHandler(default_inference_handler.DefaultInferenceHandler):

        def default_model_fn(self, model_dir, context=None):
            """Bir model yükler. PyTorch için, bir modeli yüklemek için varsayılan bir işlev sağlanamaz.
            Kullanıcılar, özel bir model_fn() işlevi sağlamalıdır.

            Args:
                model_dir: modelin kaydedildiği bir dizin.
                context (obj): istek bağlamı (varsayılan: None).

            Returns: Bir PyTorch modeli.
            """
            raise NotImplementedError(textwrap.dedent("""
            Lütfen bir model_fn uygulaması sağlayın.
            model_fn hakkında belgeler için https://github.com/aws/sagemaker-python-sdk adresine bakın.
            """))

        def default_input_fn(self, input_data, content_type, context=None):
            """JSON, CSV ve NPZ formatlarını işleyebilen varsayılan bir input_fn.

            Args:
                input_data: içerik türü formatında seri hale getirilmiş istek verisi
                content_type: istek içerik türü
                context (obj): istek bağlamı (varsayılan: None).

            Returns: torch.FloatTensor veya cuda kullanılabilirse torch.cuda.FloatTensor formatına dönüştürülmüş input_data.
            """
            return decoder.decode(input_data, content_type)

        def default_predict_fn(self, data, model, context=None):
            """PyTorch için varsayılan bir predict_fn. input_fn tarafından deserialize edilen veri üzerinde bir modeli çağırır.
            Tahminlemeyi cuda kullanılabilirse GPU üzerinde çalıştırır.

            Args:
                data: input veri (torch.Tensor), input_fn tarafından deserialize edilmiştir
                model: model_fn tarafından belleğe yüklenen PyTorch modeli
                context (obj): istek bağlamı (varsayılan: None).

            Returns: bir tahmin
            """
            return model(input_data)

        def default_output_fn(self, prediction, accept, context=None):
            """

PyTorch için varsayılan bir output_fn. predict_fn tarafından elde edilen tahminleri JSON, CSV veya NPY formatına serialize eder.

            Args:
                prediction: predict_fn tarafından elde edilen bir tahmin sonucu
                accept: çıktı verisinin serialize edilmesi gereken tür
                context (obj): istek bağlamı (varsayılan: None).

            Returns: serialize edilmiş çıktı verisi
            """
            return encoder.encode(prediction, accept)
    ```
    Not: Çıkarım işleyici fonksiyonlarına argüman olarak bağlam (context) geçmek isteğe bağlıdır. İstemci, bağlamı işlev bildiriminden çıkarabilirse çalışma zamanında kullanmayı tercih edebilir. Örneğin, aşağıdaki işleyici fonksiyon bildirimleri de işe yarar olacaktır:

    ```
    def default_model_fn(self, model_dir)

    def default_input_fn(self, input_data, content_type)

    def default_predict_fn(self, data, model)

    def default_output_fn(self, prediction, accept)
    ``` 

2.  Model sunucusu tarafından yürütülen bir işleyici hizmeti uygulayın.
    ([İşte bir örnek](https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/handler_service.py) bir işleyici hizmeti örneği.)
    `HANDLER_SERVICE` dosyasını nasıl tanımlayacağınız hakkında daha fazla bilgi için [MMS özel hizmet belgelerine](https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md) bakın.

    ``` python
    from sagemaker_inference.default_handler_service import DefaultHandlerService
    from sagemaker_inference.transformer import Transformer
    from sagemaker_pytorch_serving_container.default_inference_handler import DefaultPytorchInferenceHandler


    class HandlerService(DefaultHandlerService):
        """Model sunucusu tarafından yürütülen bir işleyici hizmeti.
        Kullanılan modelle ilgili belirli varsayılan çıkarım işleyicilerini belirler.
        Bu sınıf, aşağıdakileri tanımlayan ``DefaultHandlerService``'den türetilir:
            - ``handle`` yöntemi, model sunucusuna gelen tüm giriş çıkarım istekleri için çağrılır.
            - ``initialize`` yöntemi, model sunucusu başlatıldığında çağrılır.
        Kaynak: https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md
        """
        def __init__(self):
            transformer = Transformer(default_inference_handler=DefaultPytorchInferenceHandler())
            super(HandlerService, self).__init__(transformer=transformer)
    ```

3.  Model sunucusunu başlatan bir hizmet giriş noktası uygulayın.
    ([İşte bir örnek](https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/serving.py) bir hizmet giriş noktası örneği.)

    ``` python


    from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
    from sagemaker_pytorch_serving_container.default_handler_service import DefaultHandlerService


    def main():
        handler_service = DefaultHandlerService()
        handler_service.initialize()
        handler = DefaultInferenceHandler()
        handler_service.handle(data, context)
    ```

4.  Model sunucusunu Docker konteynerinde çalıştırın ve etkinleştirin.
    ([İşte bir örnek](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/multi_model_bring_your_own/container/dockerd-entrypoint.py) bir Docker giriş noktası örneği.)

    ``` python
    import sys

    from sagemaker_inference import log, utils

    from sagemaker_pytorch_serving_container import handler_service, serving
    from sagemaker_pytorch_serving_container.default_inference_handler import DefaultInferenceHandler


    if __name__ == "__main__":
        if len(sys.argv) < 2 or sys.argv[1] not in ("serve", "train"):
            raise ValueError(
                'Invalid argument. Please use "serve" or "train" as the first argument.'
            )

        if sys.argv[1] == "serve":
            # Serve the model
            serving.main()
        elif sys.argv[1] == "train":
            # Train the model
            pass
    ```

    Not: `if __name__ == "__main__":` kontrolü ile `train` veya `serve` komutlarına dayalı olarak modelin eğitimi veya sunulması sağlanır.

Bu adımları uyguladıktan sonra, model sunucusu Docker konteynerinde çalışacak ve SageMaker tarafından çağrılan çıkarım işleyicileri ve hizmetlerini kullanacaktır.

Umarım bu bilgiler yararlı olur! Başka sorularınız varsa, yardımcı olmaktan mutluluk duyarım.