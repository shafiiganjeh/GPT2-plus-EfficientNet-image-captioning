import tensorflow as tf
# import sys
# sys.path.append(".")
from model import models as md

def create_model(efficient_net = None,hparams = None,n_spe = 1,img_size = (250,250),encoder_layers = 2,encoder_head = 12):
    
    inp_x = tf.keras.Input(shape=(None,),dtype=tf.int32)
    inp_img = tf.keras.Input(shape=(*img_size,3))
    enc_inp = efficient_net(inp_img)
    
    if enc_inp.shape[1]*enc_inp.shape[2] > 1024:
        import warnings
        warnings.warn("The Latent image dimension seems to be very large, you might want to reduce the image size of your efficient net model.")
        input("Press Enter to continue...")
    
    enc_ = md.encoder(n_ctx = enc_inp.shape[1]*enc_inp.shape[2],train = True,n_embd = hparams["n_embd"],
                      n_layer = encoder_layers,n_head = encoder_head)
    
    enc_outp = enc_(enc_inp)
    enc = tf.keras.Model(inputs=[inp_img], outputs=[enc_outp])
    
    decoder = md.model(cross = True,emb_train = False, n_spe = n_spe,train = True,
                    n_vocab = hparams["n_vocab"],n_ctx = hparams["n_ctx"],
                    n_embd = hparams["n_embd"],n_head = hparams["n_head"],
                    n_layer = hparams["n_layer"])
    
    dec_out = decoder(x = inp_x, y = enc_outp, past = None)
    model_final = tf.keras.Model(inputs=[inp_img,inp_x], outputs=[dec_out])
    
    return enc,decoder,model_final
