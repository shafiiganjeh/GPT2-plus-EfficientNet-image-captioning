import tensorflow as tf
import tensorflow_models as tfm
from tensorflow.keras import backend as K
import sys
sys.path.append("..")
import layers as l

class model(tf.keras.Model):
    def __init__(
        self,
        n_ctx = 1024,
        n_embd = 1024,
        n_vocab=50257,
        train = True,
        cross = False,
        emb_droput = 0,
        n_spe = 0,
        n_layer = 24,
        mdrop = .1,
        pdrop = .1,
        rdrop = .1,
        n_head = 16,
        mask = True,
        emb_train = True
    ):
        super().__init__()
        
        self.past = tf.zeros([1, n_layer, 2, n_head, 1, int(n_embd/n_head)],dtype=tf.float32)
        self.n_layer = n_layer
        self._block = [None]*n_layer
        self.n_embd = n_embd
        self.n_vocab = n_vocab+n_spe
        self.train = train
        
        self.wpe = self.add_weight("wpe", 
                                     shape = [n_ctx, n_embd], 
                                     initializer = tf.random_normal_initializer(stddev=0.01, seed=None),
                                     trainable = emb_train)
        
        self.wte = self.add_weight("wte", 
                                     shape = [n_vocab, n_embd], 
                                     initializer = tf.random_normal_initializer(stddev=0.01, seed=None),
                                     trainable = emb_train)
        
        self.wsp = self.add_weight("wsp", 
                                     shape = [n_spe, n_embd], 
                                     initializer = tf.random_normal_initializer(stddev=0.01, seed=None),
                                     trainable = True)
        
        self.lm_drop = tf.keras.layers.Dropout(emb_droput)
        
        for i in range(self.n_layer ):
            self._block[i] = l.block(train = train,n_head = n_head,
                          mdrop = mdrop,pdrop = pdrop,mask = mask,
                          rdrop = rdrop,scale = True,
                          name_ = "h"+str(i),cross = cross)
            
        self.ln_f = l.norm(name_ = 'ln_f')

        

    def call(self, x,past = None,y = None):
        results = {}
        x_shape = tf.shape(x)
        past_length = 0 if past is None else tf.shape(past)[-2]
        wte = tf.keras.layers.Concatenate(axis = 0)([self.lm_drop(self.wte),self.wsp])
        wpe = self.lm_drop(self.wpe, training=self.train)
        
        h = tf.gather(wte, x) + tf.gather(wpe, tf.tile([tf.range(x_shape[1]) + past_length],[x_shape[0],1]))
        
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.n_layer
        
        for layer, past in enumerate(pasts):
            h, present = self._block[layer](x = h,past = past,y = y)
            presents.append(present)
            
        # results['present'] = None
        results['present'] = tf.stack(presents, axis=1)
        h = self.ln_f(h)
        
        h_flat = tf.reshape(h, [x_shape[0]*x_shape[1], self.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [x_shape[0], x_shape[1], self.n_vocab])
        results['logits'] = logits
        return  results
    

class encoder(tf.keras.Model):
    def __init__(
        self,
        n_ctx = 169,
        n_embd = 768,
        n_special = 0,
        n_head = 12,
        mask = False,
        n_layer = 8,
        pdrop = .1,
        rdrop = .1,
        mdrop = .1,
        train = True,
        scale = True,
        
    ):
        super().__init__()
        self.train = train  
        self.n_head = n_head 
        self.n_ctx = n_ctx
        self.n_layer = n_layer
        self.mask = mask

        
        self._block = [None]*self.n_layer
        
        self.pre_emb = tf.keras.layers.Conv1D(n_embd,1)
        
        self.norm_inp = tf.keras.layers.GroupNormalization()

        self.pos = tfm.nlp.layers.PositionEmbedding(n_ctx+2)

        for i in range(self.n_layer ):
            self._block[i] = l.block(train = self.train,n_head = self.n_head,
                          mdrop = mdrop,pdrop = pdrop,mask = self.mask,
                          rdrop = rdrop,scale = scale,
                          name_ = "block_"+str(i))
            

    def call(self, x):
        
        h = x
        h = tf.keras.layers.Reshape((-1, h.shape[-1]))(h)
        
        h = tf.pad(h,[[0, 0],[1, 1],[0, 0]], constant_values=1/h.shape[-1])  
        
        h = self.norm_inp(h)
        h = self.pre_emb(h)

        h = self.pos(h) + h
        
        for i in range(self.n_layer):
            h, present = self._block[i](x = h, past = None,y = None)
            
        h = h[:,1:-1,:]
        
        return h
    # return h,mask
    
    