import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import tensorflow as tf
import nvidia.dali.plugin.tf as dali_tf
import sys
from trainer import metrics
sys.path.append("..")
from tools import encoder

@tf.function(jit_compile=True)
def train_step(inp,model_final,optimizer,batch_size,seq_max,past):

    with tf.GradientTape(persistent = False ) as tape:

      H = model_final((inp[0],inp[1],past ),training=True) 
    
      # Compute the loss value for this minibatch.
      l1 = metrics.lm_loss(inp[1],H[0]['logits'],inp[2])
      train_loss = l1
    
    grads = tape.gradient(train_loss , model_final.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_final.trainable_weights))
    acc = metrics.calculate_accuracy(inp[1], H[0]['logits'], inp[2],batch_size = batch_size,seq_max = seq_max )
        
    return train_loss,acc


def train_ds(batch_size = 32,seq_max = 70,start = 50257,end = 50256,dataset_list = None,
             image_label = "jpg",txt_label = "txt",shuffle  = True,encoder = None,img_size = (256,256)):
    
    if encoder.__class__.__name__ != 'Encoder':
        encoder = encoder.dummy_tok(encoder)
    
    try:
        pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
        with pipe:
            im,txt = fn.readers.webdataset(
                paths=dataset_list,
                ext=[image_label, txt_label],
                random_shuffle=shuffle,
                missing_component_behavior="skip")
            images = fn.decoders.image(im, device="cpu", output_type=types.RGB)
            images = ops.Cast(device="cpu", dtype=types.FLOAT)(images)
            resized = fn.resize(images, resize_x =img_size[0], resize_y =img_size[1])
            
        pipe.set_outputs(resized,txt)
    
        with tf.device('/cpu:0'):
            ds = dali_tf.DALIDataset(
                pipeline=pipe,
                batch_size=1,
                # output_shapes=shapes,
                output_dtypes=(tf.float32,tf.uint8),
                device_id=0)
            
        @tf.autograph.experimental.do_not_convert
        def std(x,y):
            x = tf.squeeze(x)
            y = tf.squeeze(y)
            y = tok(txt = y, start = start,end = end,seq_max = seq_max,enc = encoder)
            return x,y
            
        dss = ds.map(lambda x,y: tf.py_function(func=std,inp=[x,y], Tout=(tf.float32,tf.int32))).batch(batch_size)  
        return dss
    finally:
        def tok(txt, start = start,end = end,seq_max = seq_max,enc = encoder):
            # for el in txt:
            try:
                string = "".join([chr(item) for item in txt])
            except:
                string = "A photo of"
            tok = enc.encode(string)
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


def train(steps,optimizer,ds,model):
    
    batch = next(iter(ds))
    past = model._past_
    past = tf.repeat(past,repeats=batch[1].shape[0], axis=0)
    
    l_cum = np.zeros(25)
    l_acc = np.zeros(25)
    
    for step, dali in enumerate(ds):
        
        a,b = tf.unstack(dali[1],axis = 1)
        x_batch = [dali[0],a,b]
        
        l1,acc = train_step(x_batch,optimizer = optimizer,model_final = model,batch_size = x_batch[1].shape[0],seq_max = x_batch[1].shape[1],past = past)
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

