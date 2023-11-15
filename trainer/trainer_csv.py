import os
import numpy as np
import tensorflow as tf
import sys
from math import floor
from trainer import metrics
import random

def tok(txt, start = 50257,end = 50256,seq_max = 1,enc = None):
    # for el in txt:
    tok = enc.encode(txt)
    tok = [start] + tok + [end]
    mask = [1]*len(tok)
    l = seq_max-len(tok)
    mask = mask + [0]*l
    tok = tok + [end]*l
    if len(tok) > seq_max:
        mask = mask[:seq_max]
        tok = tok[:seq_max]
    tok = np.array(list(tok),dtype="int32")
    return tok, np.array(mask)

@tf.function(jit_compile=True)
def train_step(inp,model_final,optimizer,batch_size,seq_max):

    with tf.GradientTape(persistent = False ) as tape:

      H = model_final((inp[0],inp[1]) ,training=True) 
    
      # Compute the loss value for this minibatch.
      l1 = metrics.lm_loss(inp[1],H[0]['logits'],inp[2])
      train_loss = l1

    grads = tape.gradient(train_loss , model_final.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_final.trainable_weights))
    acc = metrics.calculate_accuracy(y_true = inp[1], y_pred = H[0]['logits'], mask = inp[2],batch_size = batch_size,seq_max = seq_max )
        
    return train_loss,acc


class CSVInputIterator(object):
    def __init__(self, batch_size, images_folder, csv_df, shuffle=True,text_column = 'caption',
                 file_column = 'file_name',image_size = (256,256), start = 50267,
                 end = 50256,seq_max = 1,enc = None):
        self.images_folder = images_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.csv = csv_df

        self.data_set_len = len(self.csv)

        self.file_column = file_column
        self.text_column = text_column
        self.image_size = image_size
        
        self.start = start
        self.end = end
        self.seq_max = seq_max
        self.enc = enc

    def __iter__(self):
        order = list(range(len(self.csv)))
        if self.shuffle:
            random.shuffle(order)

        for idx in order:
            filename = self.csv[self.file_column][idx]
            tokens = tok(txt = self.csv[self.text_column][idx], start = self.start,end = self.end,seq_max = self.seq_max,enc = self.enc)
            label1 = tokens[0] 
            label2 = tokens[1] 

            
            labels1 = np.array(label1, dtype=np.int32)
            labels2 = np.array(label2, dtype=np.int32)
            
            img = tf.io.read_file(os.path.join(self.images_folder, filename))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.image_size)
            # img = tf.image.flip_left_right(img)
            img = tf.image.convert_image_dtype(img, tf.float32)

            yield (img, labels1, labels2)

    @property
    def size(self):
        return self.data_set_len


def train_ds(batch_size = 32,seq_max = 70,start = 50257,end = 50256,folder = "",csv = "",encoder = None,text_column = "",file_column = "", image_size = (256,256),shuffle = True):
    
    
    csvii = CSVInputIterator(batch_size, folder, shuffle=shuffle,csv_df = csv,image_size = image_size,text_column = text_column,
                             file_column = file_column , start = start,end = end,seq_max = seq_max,enc = encoder)
    max_len = csvii.size
    max_len = max_len/batch_size
    
    ds = tf.data.Dataset.from_generator(
        csvii.__iter__,
        output_types=(tf.float32,tf.int32,tf.int32)).batch(batch_size)
    
    return ds,floor(max_len)


def train(steps,optimizer,ds,model,max_steps):
        
    l_cum = np.zeros(25)
    l_acc = np.zeros(25)
    
    for step, x_batch in enumerate(ds):
        
        l1,acc = train_step(inp = x_batch,optimizer = optimizer,model_final = model,batch_size = x_batch[1].shape[0],seq_max = x_batch[1].shape[1])
        # if nan:
        #     break
        l_acc[0] = acc
        l_cum[0] = tf.reduce_mean(l1)
        l_cum = np.roll(l_cum, 1)
        l_acc = np.roll(l_acc, 1)
            
        sys.stdout.write('\r'+("step: " +str(step+1) + "/" + " loss: " + str(np.sum(l_cum)/25)+ " acc: " + str(np.sum(l_acc)/25) + " memory: " + str(round(tf.config.experimental.get_memory_info('GPU:0')["peak"]*1e-6)) + "   "))
        if step > steps:
            break
    
    return
