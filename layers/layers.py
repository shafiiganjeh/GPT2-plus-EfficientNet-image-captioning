import tensorflow as tf


class norm(tf.keras.layers.Layer):
  def __init__(self, 
               axis = [-1],
               e = 1e-10,
               name_ = None
               ):
    super(norm, self).__init__()
    self.axis = axis
    self.e = e
    self.name_ = name_

  def build(self, x):
      n_state = x[-1]
      self.g = self.add_weight("g/"+str(self.name_), shape = [n_state], initializer=tf.constant_initializer(1))
      self.b = self.add_weight("b/"+str(self.name_), shape = [n_state], initializer=tf.constant_initializer(0))

  def call(self, x):
      u = tf.math.reduce_mean(x, axis = self.axis, keepdims=True)
      s = tf.math.reduce_mean(tf.square(x-u), axis = self.axis, keepdims=True)
      x = (x - u) * tf.math.rsqrt(s + self.e)
      return x*self.g + self.b


class conv1d(tf.keras.layers.Layer):
    def __init__(
        self,
        nf,
        w_init = tf.random_normal_initializer(stddev=0.02),
        b_init = tf.constant_initializer(0),
        pad = 'VALID',
        train = False,
        name_ = None
    ):
        super(conv1d, self).__init__()
        self.nf = nf
        self.w_init = w_init
        self.b_init = b_init
        self.pad = pad
        self.train = train
        self.name_ = name_

    def build(self, x):
        self.nx = x[-1]
        self.w = self.add_weight("w/"+str(self.name_), 
                                 shape = [1, self.nx, self.nf], 
                                 initializer = self.w_init,trainable = self.train)
        self.b = self.add_weight("b/"+str(self.name_), 
                                 shape = [self.nf], 
                                 initializer = self.b_init,trainable = self.train)
        
    def call(self, x):
        if self.train:
            c = tf.einsum('ble,reo->blo', x, self.w)+self.b
        else:
            x_s = tf.shape(x)[:-1]
            x = tf.reshape(x, [-1, self.nx])
            w = tf.reshape(self.w, [-1, self.nf])
            m = tf.matmul(x, w)+self.b
            c = tf.reshape(m, tf.concat([x_s,[self.nf]],axis = 0))
        return c


class MHA(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        pdrop = 0.0,
        rdrop = 0.0,
        scale = True,
        train = True,
        name_ = None,
        cross = False,
        mask = True,
        seed = 123,
        **kwargs,
    ):
        super(MHA,self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.pdrop = pdrop
        self.rdrop = rdrop
        self.scale = scale
        self.train = train
        self.mask_ = mask
        self.cross = cross
        self.name_ = name_
        self.seed = seed

        assert self.key_dim % self.num_heads == 0, "K/Q dimension not divisible by number of heads"
        
    def build(self, x, y = None):
        
        self.rs = tf.keras.layers.Reshape((-1,self.num_heads,int(self.key_dim/self.num_heads)))
        self.rs_merge = tf.keras.layers.Reshape((-1,self.key_dim))
        
        if self.cross:
            self.conv_cross_inp = conv1d(nf = self.key_dim*2, train = self.train,name_ = self.name_+str("_cross_inp"))
            self.conv_cross_out = conv1d(nf = self.key_dim, train = self.train,name_ = self.name_+str("_cross_out"))
        else:
            self.conv_inp = conv1d(nf = self.key_dim*3, train = self.train,name_ = 'c_attn/' + self.name_) 
            self.conv_out = conv1d(nf = self.key_dim, train = self.train,name_ ='c_proj/' + self.name_)
            
    def attention_mask(self,nd, ns, *, dtype):

        i = tf.range(nd)[:,None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)
    
    def mask_attn_weights(self,w):
        ns = tf.shape(w)[-1]
        nd = tf.shape(w)[-2]
        b = self.attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w    
    
    def merge_heads(self,x):
        # Reverse of split_heads
        return self.rs_merge(tf.transpose(x, [0, 2, 1, 3]))

    def multihead_attn(self,q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))
        if self.mask_:
            w = self.mask_attn_weights(w)
        w = tf.nn.softmax(w)
        w = self.dropout(w,drop = self.pdrop )
        a = tf.matmul(w, v)
        return a
    
    def dropout(self, x,drop):
        if self.train and drop > 0:
            x = tf.nn.dropout(x, drop)
        return x
    
    def split_heads(self,x):
        x = self.rs(x)
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, x, y = None,past = None):
        if self.cross:
            q_c = x
            y = tf.cast(y,x.dtype)
            y = self.conv_cross_inp(y)
            k_c, v_c = tf.split(y, 2, 2)
            
            k_c = self.split_heads(k_c)
            v_c = self.split_heads(v_c)
            q_c = self.split_heads(q_c)
            
            a2 = self.multihead_attn(q_c, k_c, v_c)
            a2 = self.merge_heads(a2) 
            a2 = self.conv_cross_out(a2)
            a = self.dropout(a2,drop = self.rdrop )
            present = None
        else:
            c = self.conv_inp(x)
            q, k, v = tf.split(c, 3, axis=2) 
            q = self.split_heads(q)
            k = self.split_heads(k)
            v = self.split_heads(v)
            
            if past is not None:
                pk, pv = tf.unstack(past, axis=1)
                k = tf.concat([pk, k], axis=-2)
                v = tf.concat([pv, v], axis=-2)
            present = tf.stack([k, v], axis=1)
                
            a = self.multihead_attn(q, k, v)
            a = self.merge_heads(a)
            a = self.conv_out(a)
            a = self.dropout(a,drop = self.rdrop )
        
        return a,present


class mlp(tf.keras.layers.Layer):
    def __init__(
        self,
        n_state,
        train = True,
        mdrop = 0.,
        name_ = None,
    ):
        super(mlp, self).__init__()
        self.train = train
        self.n_state = n_state
        self.drop = mdrop
        self.name_ = name_

    def build(self, x):
        self.nx = x[-1]

        self.c_fc = conv1d(nf = self.n_state, train = self.train,name_ = str("c_fc/")+self.name_)
        self.c_proj = conv1d(nf = self.nx, train = self.train,name_ = str("c_proj/")+self.name_)

    def dropout(self, x, pdrop, train):
        if self.train and pdrop > 0:
            x = tf.nn.dropout(x, pdrop)
        return x
        
    def call(self, X):
        h = tf.keras.activations.gelu(self.c_fc(X))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, self.drop, self.train)
        return h2
    
    
class block(tf.keras.layers.Layer):
  def __init__(self, 
               train = True,
               n_head = 12,
               pdrop = .0,
               rdrop = .0,
               mdrop = .0,
               mask = True,
               cross = False,
               scale = True,
               name_ = None,
               ):
    super(block, self).__init__()
    self.train = train
    self.pdrop = pdrop
    self.rdrop = rdrop
    self.cross = cross
    self.mdrop = mdrop
    self.scale = scale
    self.n_head = n_head
    self.name_ = name_
    self.mask = mask

    
  def build(self, x):
      nx = x[-1]
      self._mha = MHA(pdrop = self.pdrop, rdrop = self.rdrop,
                      key_dim = nx, num_heads = self.n_head, 
                      train = self.train, scale = self.scale, 
                      mask = self.mask,cross = False,
                      name_ = str('attn/') + self.name_)
      if self.cross:
          self.norm1_cross = norm(name_ = str("cross_norm1_") + self.name_)
          self._mha_cross = MHA(pdrop = self.pdrop, rdrop = self.rdrop,
                          key_dim = nx, num_heads = self.n_head, 
                          train = self.train, scale = self.scale, 
                          mask = False,cross = True,
                          name_ = str("cross_MHA_") + self.name_)
      
      self.norm1 = norm(name_ = str("ln_1/") + self.name_)
      
      self._mlp = mlp(n_state = nx*4, train = self.train,mdrop = self.mdrop,name_ = str('mlp/') + self.name_)
      
      self.norm2 = norm(name_ = str("ln_2/") + self.name_)

  def call(self, x, y = None,past = None):
      if self.cross:
          a, present = self._mha(self.norm1(x),past = past)
          x = x + a
          c,_ = self._mha_cross(x = self.norm1_cross(x),y = y,past = None)
          x = x + c
          m = self._mlp(self.norm2(x))
          x = x + m
      else:
          a, present = self._mha(x = self.norm1(x),past = past)
          x = x + a
          m = self._mlp(self.norm2(x))
          x = x + m
      return x,present

    
