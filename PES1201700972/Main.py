from Model_Utils import *
from Dataset import *
import tensorflow as tf
from train import *
saver = tf.train.Saver()
decoded_words = []
star_text = '<START> '
image = load_val_images('try/')[0]
img_tensor=np.expand_dims(np.array(image),0)
img_tensor=np.array(img_tensor)
predicted='<START>'
star_text = '<START>'
with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    for di in range(1000):
        sequence = test_dataset.tokenizer.texts_to_sequences([star_text])
        decoder_input = np.array(sequence).reshape(-1,1)
        print(decoder_input)
        a = sess.run(output_gru2, feed_dict={im:img_tensor,caption_p:decoder_input})
        print(a)
        data=list(a[0][0])
        i=data.index(max(data))
        word = word_for_id(i,test_dataset.tokenizer)
        if word is None:
            continue
        predicted += word + ' '
        star_text = word
        print(predicted)
        if word == '<END>':
            pass

