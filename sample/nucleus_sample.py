import tensorflow as tf
from tqdm import tqdm


def argmax(logits):
	return tf.argmax(logits)


def top_k_logits(logits, k):
	if k == 0:
		return logits

	values, _ = tf.nn.top_k(logits, k=k)
	min_values = values[:, -1]

	return tf.where(
		logits < min_values,
		tf.ones_like(logits, dtype=logits.dtype) * -1e10,
		logits
	)


def top_p_logits(logits, p):
	"""Took from OpenAI GPT-2 Implememtation"""
	batch = tf.shape(logits)[0]
	sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
	cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
	indices = tf.stack([
		tf.range(0, batch),
		tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
	], axis=-1)
	min_values = tf.gather_nd(sorted_logits, indices)
	return tf.where(
		logits < min_values,
		tf.ones_like(logits) * -1e10,
		logits,
	)



def sample_sequence(model,context,encoder,
						seq_len=512,
						bos=3,
						eos=4,
						temperature=1,
						top_k=8,
						top_p=8,
						nucleus_sampling=True):
    
    context = encoder(context)
    bos = tf.expand_dims(([bos]), 0)
    prev = bos
    output = bos
    past = model.past
    
    for i in tqdm(range(seq_len)):
        res = model(x = prev,y = context, past=past)
        logits = res['logits']
        past = res['present']
        logits = logits[:, -1, :] / tf.cast(temperature, tf.float32)
        logits = top_k_logits(logits, k=top_k)

        if nucleus_sampling:
            logits = top_p_logits(logits, p=top_p)

        samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
    
        if tf.equal(samples, eos):
            break

        output = tf.concat([output, samples], axis=-1)
        prev = samples
        
        # break

    result = tf.squeeze(output, axis=0)
    pred = [int(i) for i in result]

    return pred[1:]
    
    