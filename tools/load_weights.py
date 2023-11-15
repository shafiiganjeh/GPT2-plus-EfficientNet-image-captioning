import tensorflow as tf

def load(model,ckp_path = ""):
    reader = tf.train.load_checkpoint(ckp_path)
    shape_from_key = reader.get_variable_to_shape_map()
    
    c = 0

    decoder = model
    for i,weights in enumerate(decoder.weights):
        name = weights.name[:-2]
        name = name.split("/")
        # print(name)
        for c_point in shape_from_key:
            c_var = c_point
            c_point = c_point[6:].split("/")
            if set(c_point).issubset(name): 
                decoder.weights[i].assign(reader.get_tensor(c_var))
                c = c +1

    print("weights assigned: " + str(c) + "/" + str(len(shape_from_key)))  
    
    return
