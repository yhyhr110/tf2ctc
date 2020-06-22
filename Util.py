import numpy as np
import PIL
import tensorflow as tf


class CommonUtil:

    @staticmethod
    def depoint(img):  # 输入灰度化后的图片
        '''对于像素值>245的邻域像素，判别为属于背景色，如果一个像素上下左右4各像素值有超过2个像素属于背景色，那么该像素就是噪声'''
        pixdata = img.load()
        w, h = img.size
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                count = 0
                if pixdata[x, y - 1] > 245:
                    count = count + 1
                if pixdata[x, y + 1] > 245:
                    count = count + 1
                if pixdata[x - 1, y] > 245:
                    count = count + 1
                if pixdata[x + 1, y] > 245:
                    count = count + 1
                if count > 2:
                    pixdata[x, y] = 255
            return img

    @staticmethod
    def text2vec(char_set, text):
        vec = np.zeros([len(text), len(char_set)])
        for i, c in enumerate(text):
            idx = char_set.index(c)
            vec[i][idx] = 1.0
        return vec

    @staticmethod
    def seq2sparse(sequence, dtype=np.int32):
        '''序列转为稀疏矩阵'''
        indices = []
        values = []
        for i, seq in enumerate(sequence):
            indices.extend(zip([i] * len(seq), range(len(seq))))
            values.extend(seq)
        indices = np.asarray(indices, np.int64)
        values = np.asarray(values, dtype)

        shape = np.asarray([len(sequence), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    @staticmethod
    def sparse2tensor(spars_tensor):
        '''稀疏矩阵转为序列'''
        decoded_indexes = list()
        current_i = 0
        current_seq = []

        def decode_a_seq(indexes, spars_tensor):
            decoded = []
            for m in indexes:
                str = [spars_tensor[m]]
                decoded.append(str)
            return decoded

        for offset, i_and_index in enumerate(spars_tensor[0]):
            i = i_and_index[0]
            if i != current_i:
                decoded_indexes.append(current_seq)
                current_i = i
                current_seq = list()
            current_seq.append(offset)
        decoded_indexes.append(current_seq)
        result = []
        for index in decoded_indexes:
            result.append(decode_a_seq(index, spars_tensor[1]))
        return result
