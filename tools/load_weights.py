import tensorflow as tf
import h5py

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

def str_transform(string):
    
    string = string.replace('_._', '').replace('bias', 'b').replace('gamma', 'g').replace('beta', 'b').replace('wte/weight', 'wte').replace('weight', 'w').replace('embeddings', '').replace(':0', '')
    return string


def read_hdf5(path):

    weights = {}
    
    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                weights[str_transform(f[key].name)] = f[key][()]
    return weights

def load_hdf5(model,hdf5_path = ""):
    c = 0

    decoder = model
    hdf5 = read_hdf5(hdf5_path)
    
    s = ""
    for i in decoder.weights:
        i = i.name
        for j in range(10):
            i = i.replace(':'+str(j), '')
        s = s + "/" + i
    
    s = s.split("/")
    s = set(s)
    
    for i,weights in enumerate(decoder.weights):
        name = weights.name[:-2]
        name = (name + "/").split("/")
        # print(name)
        for c_point in hdf5:
            c_var = c_point
            c_point = c_point.split("/")
            if (set(c_point).intersection(s)).issubset(name): 
                # print(name)
                # print(c_point)
                try:
                    decoder.weights[i].assign(hdf5[str(c_var)])
                except:
                    try:
                        decoder.weights[i].assign(tf.expand_dims(hdf5[str(c_var)],axis = 0))
                    except:
                        decoder.weights[i].assign(tf.squeeze(hdf5[str(c_var)],axis = [0]))
                print(c)
                c = c +1

    print("weights assigned: " + str(c) + "/" + str(len(hdf5)))

    return 
