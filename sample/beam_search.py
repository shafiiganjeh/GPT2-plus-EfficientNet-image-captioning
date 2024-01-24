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


def step(logits,beams,output,probs,eos,output_probs,output_tok):
    
    with tf.device('/CPU:0'):
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

    return beams,output,probs,p_mask,output_probs,output_tok



def beam_sample(model,context,encoder,hparams,
						seq_len=15,
						bos=3,
						eos=4,
                        beams =3,
                        batch = 3
                        ):
    
    output_probs = [[] for i in range(batch)]
    output_tok = [[] for i in range(batch)]
    
    context = encoder(context)
    
    bos = tf.expand_dims(([bos]), 0)
    prev = tf.repeat(bos,beams*batch,axis = 0)
    output = tf.repeat(bos,1,axis = 0).numpy()
    
    try:
        past = tf.zeros((1, hparams["n_layer"], 2, hparams["n_head"], 1, int(hparams["n_embd"]/hparams["n_head"])))
    except:
        past = model.past
        
    past = tf.repeat(past,beams*batch,axis = 0)
    
    reps = [beams]*batch
    y = tf.repeat(context,reps,axis = 0)
    
    probs = np.ones(beams*batch,np.float64)
    
    beam_batch = [beams]*batch
    output_batch = [output]*batch
    probs_batch = [probs]*batch
    p_mask = [[] for i in range(batch)]
    
    
    for i in tqdm(range(seq_len)):
        
        if sum(beam_batch) > 0:
        
            with tf.device("GPU:0"):
                
                prev = tf.cast(prev, tf.int32)
                res = model(x = prev,y = y, past=past)
                
                res_split = tf.split(res['logits'],[i for i in beam_batch])
                present_split = tf.split(res['present'],[i for i in beam_batch])
                
            
            for b in range(batch):
                # print("batch" + str(b)+str("beams:")+str(beam_batch[b]))
                if beam_batch[b] > 0:
                    beam_batch[b],output_batch[b],probs_batch[b],p_mask[b],output_probs[b],output_tok[b] = step(
                        logits = res_split[b].numpy(),
                         beams = beam_batch[b],
                         output = output_batch[b],
                         probs = probs_batch[b],eos = eos,
                         output_probs = output_probs[b],
                         output_tok =output_tok[b])
                     
            past = []
            prev = []
            y = []
            
            with tf.device("GPU:0"):
                for b in range(batch):
                    
                    past.append(tf.gather(present_split[b],p_mask[b]))
                    prev.append(tf.constant(output_batch[b][:,-1:]))
                    y.append(tf.repeat(tf.expand_dims(context[b],axis = 0),beam_batch[b],axis = 0))
                
                y = tf.concat(y,axis = 0)
                prev = tf.concat(prev,axis = 0)
                past = tf.concat(past,axis = 0)

    return output_probs,output_tok 


