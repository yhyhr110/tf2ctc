import numpy as np
import tensorflow as tf
number = list(map(lambda x:str(x),range(10)))
alphabet = list(map(lambda x:chr(x),range(ord('a'),ord('z')+1)))
ALPHABET = list(map(lambda x:chr(x),range(ord('A'),ord('Z')+1)))

CHAR_SET = number+alphabet+ALPHABET

t = [['1','i'],['1','E'],['1','2','A','2']]


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []

    values = []

    for n, seq in enumerate(sequences):
        i = zip([n] * len(seq), range(len(seq)))
        indices.extend(zip([n] * len(seq), range(len(seq))))

        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)

    values = np.asarray(values, dtype=dtype)

    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def decode_sparse_tensor(a,b):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(a):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, b))
    return result

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = [spars_tensor[m]]
        decoded.append(str)
    return decoded

if __name__ == '__main__':
    # a,b,c = sparse_tuple_from(t)
    # sparse = tf.SparseTensor(a, b, c)
    # tensor = decode_sparse_tensor(a, b)
    # print(tensor)
    l = ['H', 'l', 'h', 'A', 'm', 'z', 'h', 'j']
    vec = list(map(lambda x: CHAR_SET.index(x), l))
    print(vec)
