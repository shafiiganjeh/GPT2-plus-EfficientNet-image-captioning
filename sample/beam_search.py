import tensorflow as tf
from tqdm import tqdm
import numpy as np

def softmax_stable(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

def k_sample(k,tensor,logit,prob_old):
    # tensor = tensor.numpy()
    logit = np.squeeze(logit,axis=1).astype(np.float64)
    
    prob = [None]*len(tensor)
    new_tensor = [None]*len(tensor)
    pasts = [None]*len(tensor)
    
    for step,t in enumerate(tensor):
        
        ind = np.argpartition(softmax_stable(logit[step]), -k, axis=-1)[-k:]
        new_tensor[step] = np.insert(ind.reshape((ind.size,1)),values = t.reshape((t.size,1)),axis = 1,obj = 0)
        
        prob[step] = softmax_stable(np.take_along_axis(logit[step], ind, axis=-1))*prob_old[step]
        prob[step] = prob[step].reshape((prob[step].size,1))
        pasts[step] = np.repeat(step,k)
        # break

    ind = ind.reshape((ind.size,1))
    return np.concatenate(new_tensor, axis=0 ),np.concatenate(prob, axis=0 )[:,0],np.concatenate(pasts, axis=0)


def prune(beam,prob,pasts,k):
    ind = prob.argsort()[-k:]
    beam = beam[ind,:]
    prob = prob[ind]
    past_index = pasts [ind]
    return beam,prob,past_index


def beam_sample(model,context,encoder,
						seq_len=15,
						bos=3,
						eos=4,
                        beams =3):
    
    output_probs = []
    output_tok = []
    
    context = encoder(context)
    
    bos = tf.expand_dims(([bos]), 0)
    prev = tf.repeat(bos,beams,axis = 0)
    output = tf.repeat(bos,1,axis = 0).numpy()
    past = model.past
    past = tf.repeat(past,beams,axis = 0)
    y = tf.repeat(context,beams,axis = 0)
    probs = np.ones(beams,np.float64)
    
    
    for i in tqdm(range(seq_len)):
        
        with tf.device("GPU:0"):
            res = model(x = prev,y = y, past=past)
            past = res['present']
                 
        with tf.device('/CPU:0'):
            logits = res['logits'].numpy()
            output,probs,p_mask = k_sample(beams,output,logits,probs)  
            output,probs,p_mask = prune(beam = output,prob = probs,pasts = p_mask,k = beams)

            if any(output[:,-1:] == eos):
                pos = np.where(output[:,-1:] == eos)[0]
                
                for i in pos:
                    output_probs.append(probs[i])
                    output_tok.append(output[i])
                    
                probs = np.delete(probs, pos, 0)
                p_mask = np.delete(p_mask, pos, 0)
                output = np.delete(output, pos, 0)
                beams = beams - len(pos)
                
        if beams < 1:
            break

        with tf.device("GPU:0"):
            past = tf.gather(res['present'],p_mask)
            prev = tf.constant(output[:,-1:])
            y = tf.repeat(context,beams,axis = 0)
        

    return output_probs,output_tok #output[np.argmax(probs)][1:-1]


    