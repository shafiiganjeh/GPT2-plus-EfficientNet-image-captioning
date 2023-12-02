# GPT2-plus-EfficientNet-image-captioning

GPT2-plus-EfficientNet-image-captioning lets you combine any EfficientNet model with GPT2 via cross attention to build an image captioning model. This repository also contains a csv trainer for csv/image datasets like the <a href="https://cocodataset.org/#home"> COCO</a> and a webdataset trainer build on the NVIDIA Data Loading Library for datasets like the ones acquired trough  <a href="https://github.com/rom1504/img2dataset/tree/main"> img2dataset</a> .  For sampling, there is a beam search sampler and a nucleus sampler. I made this repository because I needed an image captioning model that is highly customizable and gets good results on very little data but can also be scaled to work on large datasets like cc12m.
<br/><br/>
<p align="center">
    <img src=".//res/im.jpg" alt="drawing" style="width:320px;"/>
    <center>A baby elephant standing on top of a bicycle.</center>
</p>


## How to use:
 Quickstart on <a href="https://colab.research.google.com/drive/1PpMtwdCvtiRbrSlYAPCZB51TuI9KYYDg?usp=sharing"> google colab</a> .


You can get the GPT2 weights either from <a href="https://huggingface.co/gpt2"> 🤗</a> and use tools.load_hdf5 (config.json = hparams.json) or you can get the original weights from <a href="https://github.com/openai/gpt-2/blob/master/download_model.py"> OpenAi</a> and use tools.load.


## Effect of model size:
Depending on your dataset size, larger models are significantly better at captioning out-of-sample/context data. Here is a comparison between a small and medium model trained on the same dataset with the same training parameters.

````
Optimizer: ADAM 
Parameters :    lr = 6.25e-5,beta_1=0.9,beta_2=0.999,epsilon=1e-08,weight_decay=None, clipvalue=1,global_clipnorm=1
Dataset: cc3m 2 epochs + mscoco 1 epoch    
model.1: GPT2 124M + EfficientNetV2 S + 2 encoder layers
model.2: GPT2 345M + EfficientNetV2 M + 8 encoder layers 
Image size 400 x 400
````


| <img src=".//res/c1.jpg" alt="drawing" style="width:210px;"/> |<img src=".//res/c2.jpg" alt="drawing" style="width:360px;"/> | <img src=".//res/c4.jpg" alt="drawing" style="width:380px;"/> |
|:---:|:---:|:---:|
|m1: A woman kneeling down next to a large black and white axe.|m1 :A crab is eating a banana while sitting on a couch.|m1: A man riding a skateboard down the side of a pool.|
|m2: A woman laying on the ground next to a giant spider. |m2: A crab is on the ground with a knife.|m2: A toy car sitting on top of a toy pool. |

On regular data, both models perform fairly well, the smaller model even outperforms the larger one. So I would not recommend taking a larger model unless you need your model to deal with irregular samples and also have lots of data. :

| <img src=".//res/c3.jpg" alt="drawing" style="width:490px;"/> |<img src=".//res/c5.jpg" alt="drawing" style="width:380px;"/> | <img src=".//res/c6.jpg" alt="drawing" style="width:210px;"/> |
|:---:|:---:|:---:|
|m1: A pair of tanks are parked on a grassy field.|m1 :A group of men playing basketball on a basketball court.|m1: A group of people standing around a birthday cake with candles.|
|m2: A large army tank is parked in a field.|m2: A group of men playing a game of basketball.|m2: A group of people standing around a cake. |
## Pretrained models:

model.1 checkpoint on cc3m+coco:

<a href="https://drive.google.com/file/d/1CaiCE9TG-TIl2lEl8zQK7u9yJU6ek6RR/view?usp=sharing"> model.1 encoder </a>  
<a href="https://drive.google.com/file/d/143ACIbCT_ocSDoq0LtyRmZn7WRAJmbRq/view?usp=sharing">model.1 decoder </a> 

model.2 checkpoint on cc3m:

<a href="https://drive.google.com/file/d/1piNJ5bU7JgHF0M3k-R07pp4x9dGhV7su/view?usp=sharing"> model.2 encoder </a>  
<a href="https://drive.google.com/file/d/1FE4COJkkXolaDp4R0B-I_KNXHO235mY3/view?usp=sharing">model.2 decoder </a> 

model.2 checkpoint on cc3m+mscoco:

<a href="https://drive.google.com/file/d/1KS-xKfomTvtDvc2qX4e0kPN1KcgVbShl/view?usp=sharing"> model.2 encoder </a>  
<a href="https://drive.google.com/file/d/1Eb_sK8yP7uLOy_G-kxf8AYbZDFcZUpab/view?usp=sharing">model.2 decoder </a> 



## Some technical details:

The encoder takes the output of any EfficientNet model (minus the fully-connected layer at the end) which is a high-dimensional latent image. Then it flattens the image to a sequence and changes its dimension to that of the token embedding dimension. Then it adds a learned positional embedding. I did not add a relative positional embedding because CNNs, which are used in EfficientNet already carry relative information. The sequence is then fed into bidirectional transformer blocks. The encoder output is injected into every attention layer with K/V as the latent image sequence.

