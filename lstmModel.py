import numpy as np
import tensorflow as tf
from captcha.image import  ImageCaptcha
from PIL import Image
import random
from Util import CommonUtil

number = list(map(lambda x:str(x),range(10)))
alphabet = list(map(lambda x:chr(x),range(ord('a'),ord('z')+1)))
ALPHABET = list(map(lambda x:chr(x),range(ord('A'),ord('Z')+1)))

CHAR_SET = number+alphabet+ALPHABET
SAVE_PATH = r"E:\pycharmProjects\tf2.0lstmCTC验证码识别\models"
CHAR_SET_LEN =len(CHAR_SET)
IMAGE_SHAPE = (60,200)
num_epochs =4
learning_rate = 0.001
batch_size = 512
max_captcha_size = 8

def random_captcha_text(char_set=None,max_captcha_size=8):
    text = []
    if char_set is None :
        char_set=CHAR_SET
    captcha_size = random.randint(4,max_captcha_size)
    for i in  range(captcha_size):
        c = random.choice(char_set)
        text.append(c)
    return text

def gen_img_and_text(width=200,height=60,char_set=None):
    image = ImageCaptcha(width=width,height=height)
    captha_text = random_captcha_text(char_set)
    vec = list(map(lambda x: char_set.index(x), captha_text))
    captha_text = "".join(captha_text)
    captha = image.generate(captha_text)
    captha_image = Image.open(captha)
    captha_image = captha_image.convert('L')
    captha_image = CommonUtil.depoint(captha_image)
    captha_image = np.array(captha_image)
    return captha_image,captha_text,vec

def text2vec(text):
    vec = np.zeros([len(text), CHAR_SET_LEN])
    for i,c in enumerate(text):
        idx = CHAR_SET.index(c)
        vec[i][idx]=1.0
    return vec

def vec2text(vector):
    text = []
    for i,c in enumerate(vector):
        text.append(CHAR_SET[c])
    return "".join(text)


# 生成一个训练batch
def get_next_batch(batch_size=batch_size):
    # (batch_size,200,60)
    inputs = np.zeros([batch_size, IMAGE_SHAPE[1], IMAGE_SHAPE[0]])
    codes = []

    for i in range(batch_size):
        # 生成不定长度的字串
        image, text, vec = gen_img_and_text(char_set = CHAR_SET)
        # np.transpose 矩阵转置 (60*200,) => (60*200) => (200,60)
        inputs[i, :] = np.transpose(image.reshape((IMAGE_SHAPE[0], IMAGE_SHAPE[1])))
        # 标签转成列表保存在codes
        codes.append(vec)
    # 比如batch_size=2，两条数据分别是"12"和"1"，则targets [['1','2'],['1']]
    targets = [np.asarray(i) for i in codes]
    # targets转成稀疏矩阵
    sparse_targets = CommonUtil.seq2sparse(targets)
    # (batch_size,) sequence_length值都是200，最大划分列数
    seq_len = np.ones(inputs.shape[0]) * IMAGE_SHAPE[1]
    return inputs, sparse_targets, seq_len

def crack_captcha_lstm(allow_cudnn_kernel=True):
    if allow_cudnn_kernel:
        lstm_layer = tf.keras.layers.LSTM(100, input_shape=(batch_size,IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    else:
        lstm_layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100),input_shape=(batch_size, IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    model = tf.keras.models.Sequential([
        lstm_layer,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(20, activation='softmax')])
    return model

def train():
    try:
        model = tf.keras.models.load_model(SAVE_PATH+'\model')
    except Exception as e:
        model = crack_captcha_lstm()
    model.compile(optimizer='adma', loss=tf.raw_ops.CTCLoss(), metrices=['accuracy'])
    for times in range(5000):
        inputs, sparse_targets, seq_len= get_next_batch(512)
        print('times=', times, ' inputs.shape=', inputs.shape, ' sparse_targets.shape=', sparse_targets.shape, ' seq_len.shape=', seq_len.shape)
        model.fit(batch_x, batch_y, epochs=4)
        print("y预测=\n", np.argmax(model.predict(batch_x), axis=2))
        print("y实际=\n", np.argmax(batch_y, axis=2))

        if 0 == times % 10:
            print("save model at times=", times)
            model.save(SAVE_PATH + 'model')



if __name__ == '__main__':
    a,b,c = get_next_batch()
    tensor = CommonUtil.sparse2tensor(a[1])
    print(tensor)
