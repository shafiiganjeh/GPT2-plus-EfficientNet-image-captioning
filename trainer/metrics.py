import tensorflow as tf

# @tf.function(jit_compile=True)
def lm_loss(label, pred, mask):
    pred = pred[:,:-1,:]
    pred = tf.reshape(pred,[pred.shape[0]*pred.shape[1],-1])
    
    label = label[:,1:]
    label = tf.reshape(label,[-1])
    mask = mask[:,1:]
    mask = tf.reshape(mask,[-1])
    mask = tf.cast(mask,tf.float32)
        
    lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    lm_losses = tf.reduce_sum(lm_losses*mask)/tf.reduce_sum(mask)

    return lm_losses

# @tf.function(jit_compile=True)
def calculate_accuracy(y_true, y_pred, mask,batch_size,seq_max):
    k = seq_max -1
    y_true = y_true[:,1:]
    y_true = tf.reshape(y_true, [batch_size*k])
    mask = tf.reshape(mask[:,1:], [batch_size*k])
    mask = tf.cast(mask,tf.float32)

    y_pred = y_pred[:,:-1,:]
    y_pred = tf.reshape(y_pred,[batch_size*k,-1])
    accuracy = tf.equal(y_true, tf.cast(tf.argmax(y_pred ,axis=1),tf.int32))
    accuracy = mask*tf.cast(accuracy,tf.float32)
    return tf.math.reduce_sum(accuracy) / tf.math.reduce_sum(mask)
