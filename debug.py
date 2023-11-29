import tensorflow as tf
from model import *
from trainer import trainer_csv
import tools
import pandas as pd

inp_img = tf.keras.Input(shape=(256,256,3))
img_enc = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=False,
    weights='imagenet',
    input_tensor=inp_img,
    input_shape=None,
    pooling=None,
    include_preprocessing=True
)

img_enc.trainable = False
import json

print("Started Reading JSON file")
with open("/home/borz/Desktop/gpt2 torch/models/124M/hparams.json", "r") as read_file:
    print("Converting JSON encoded data into Python dictionary")
    developer = json.load(read_file)

optimizer = tf.keras.optimizers.Adam(
                                        learning_rate = 6.25e-5,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-10,
                                        weight_decay=None,
                                        clipvalue=1,
                                        global_clipnorm=1,
                                        jit_compile=True,
                                    )

enc,decoder,model_final = create_model(efficient_net = img_enc,hparams = developer,n_spe = 1,img_size = (256,256))

# tools.load(decoder,ckp_path = "/home/borz/Desktop/gpt2 torch/models/124M")

# mod_dir = "/home/borz/Desktop/gpt2 torch/models" 
# name = "124M"
# enc_tok = tools.get_encoder(model_name = name, models_dir = mod_dir )
# from transformers import GPT2Tokenizer, GPT2Model

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# csv_df = pd.read_csv("/media/borz/DAT/coco/annotations/com.csv",index_col=0)
# ds = trainer_csv.train_ds(batch_size = 32,seq_max = 50,start = 50257,end = 50256,folder = "/media/borz/DAT/coco/train2017",csv_df = csv_df,encoder = tokenizer,text_column = "caption",file_column = 'file_name', image_size = (256,256),shuffle = True)
# ds,l = ds
# batch = next(iter(ds))

# trainer_csv.train(steps = 100,optimizer = optimizer ,ds = ds,model = model_final)
# 
# import os
# l = []
# for path, subdirs, files in os.walk("/media/borz/DAT/cc3m"):
#     for name in files:
#         l.append(os.path.join(path, name))
        
# l = [a for a in l if ('.tar' in a[len(a)-4:])] 

# # from trainer import trainer_dail

# ds = trainer_dail.train_ds(seq_max = 50,batch_size = 32,dataset_list = l[0],encoder = enc_tok,img_size = (256,256))


# # # for i in ds.take(1): a = i


# trainer_dail.train(steps = 100,optimizer = optimizer ,ds = ds,model = model_final)

from transformers import  TFGPT2Model
model = TFGPT2Model.from_pretrained('gpt2')



