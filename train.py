from Model_Utils import *
from Dataset import *

# Making test and train datasets
dir_name = 'try/'

# test set size has been set to 1 for ease of demo
test_dataset,train_dataset = test_train_dataset(dir_name,1)
print("dataset done")

# formatting images as np array
x_train = np.array(train_dataset.images,dtype=np.float32)
for i in range(len(x_train)):
    x_train[i]=np.array(x_train[i],dtype=np.float32)

# making an np array for the expected DSLs
expected = train_dataset.y
expected=np.array(expected)
for e in range(len(expected)):
    expected[e]=np.array(expected[e])

# creating a placeholder for images
im = tf.placeholder(dtype=tf.float32, shape=(None,3,224,224), name='im')

# Creating the graph of CNN
model_train = cnn_train(im,weights,biases)
model_test = cnn_test(im,weights,biases)
output_train = batch_norm_wrapper(model_train,True)
output_test = batch_norm_wrapper(model_test,False)


# Embedding layer
VOCAB_LEN=19
EMBED_SIZE=50
embeddings = tf.Variable(tf.random_uniform([VOCAB_LEN, EMBED_SIZE]))
caption_p = tf.placeholder(dtype=tf.int32, shape=(None,None), name='caption_p')
embed = tf.nn.embedding_lookup(embeddings, caption_p)

# Encoder RNN with 2 levels
gru_before = GRU(50,256,embed)
gru_before_1 = GRU(256,256,gru_before.h_t)

# Weights for softmax layer
# Wout_gru1 = weights['Wout_gru1']
# bout_gru1 = biases['Bout_gru1']
output_gru1 = gru_before_1.h_t

# Concatenating image feature vector and embedded vector
features_try = K.tile(K.expand_dims(output_train, 1), [1, K.shape(output_gru1)[1], 1])
embeddings = tf.concat([features_try,output_gru1],2)

# Decoder GRU
gru_final = GRU(1280,512,embeddings)

Wout_gru2 = weights['Wout_gru2']
bout_gru2 = biases['Bout_gru2']

# softmax layer in Decoder
output_gru2 = tf.nn.softmax(tf.matmul(gru_final.h_t,Wout_gru2)+bout_gru2)

# Loss and optimizer
true_output = tf.placeholder(dtype=tf.float32, shape=(None,None,None), name='expected_output')
loss = tf.reduce_sum(tf.squared_difference(output_gru2 ,true_output)) #/ float(1)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# Training the model
epoch = 50
vocab_size = 19
batch_size = 20


x_train = train_dataset.images
caption = train_dataset.X
expected = train_dataset.y

saver = tf.train.Saver()

if __name__=='__main__':
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # array to store all losses across epochs
        loss_ar=[]

        for e in range(epoch):
            loss_no=[]
            print("Epoch number: ",e)
            for batch in range(len(x_train)//batch_size):
                # Generating batches
                batch_x = x_train[batch*batch_size:min((batch+1)*batch_size,len(x_train))]
                batch_y = caption[batch*batch_size:min((batch+1)*batch_size,len(caption))] 
                batch_ex = expected[batch*batch_size:min((batch+1)*batch_size,len(expected))]
                batch_x = np.array(batch_x)
                for b in range(len(batch_x)):
                    batch_x[b]=np.array(batch_x[b])
                batch_y = np.array(batch_y)
                for b in range(len(batch_y)):
                    batch_y[b]=np.array(batch_y[b])
                batch_ex = np.array(batch_ex)
                for b in range(len(batch_ex)):
                    batch_ex[b]=np.array(batch_ex[b])
                    
                
                # Padding the in sequences and out sequences
                batch_y = pad(batch_y)
                batch_ex = pad2(batch_ex)


                ls,tr = sess.run([loss,train_step],feed_dict ={true_output:batch_ex,im:batch_x,caption_p:batch_y})
                print("loss for batch no: ", batch+1," = " ,ls/batch_size)
                loss_no.append(ls/batch_size)
                print("\n\n")
            loss_ar.append(loss_no)

            print("-----------------------------------------------------------------") 
        save_path = saver.save(sess, "model.ckpt")
    
