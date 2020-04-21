from Model_Utils import *
from Dataset import *
import tensorflow as tf
from train import *
saver = tf.train.Saver()

decoded_words = []
image = test_dataset.images[0]
img_tensor=np.expand_dims(np.array(image),0)
img_tensor=np.array(img_tensor)


features_try = K.tile(K.expand_dims(output_test, 1), [1, K.shape(output_gru1)[1], 1])
embeddings = tf.concat([features_try,output_gru1],2)

predicted='<START>'
star_text = '<START>'
with tf.Session() as sess:
    saver.restore(sess, "model10.ckpt")
    for di in range(50):
        #print(star_text)
        sequence = test_dataset.tokenizer.texts_to_sequences([star_text])

        temp =[]
        for x in sequence:
            temp.append(x)
        
        temp = np.array(temp)
        print(temp.shape)
    
        a = sess.run(output_gru2, feed_dict={im:img_tensor,caption_p:temp})
        
        data=list(a[0][-1])
        i=data.index(max(data))
        word = word_for_id(i,test_dataset.tokenizer)
        if word is None:
            continue
        predicted += word + ' '
        star_text += ' ' +word
        print(predicted)
        if word == '<END>':
            break