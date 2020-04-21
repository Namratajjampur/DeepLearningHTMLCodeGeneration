import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.layers import Embedding
epsilon = 1e-3
vb_size=18
batchsize=2
weights = {
    'W_conv1': tf.get_variable('W0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_conv2': tf.get_variable('W1', shape=(3,3,32,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_conv3': tf.get_variable('W2', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_conv4': tf.get_variable('W3', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_conv5': tf.get_variable('W4', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_conv6': tf.get_variable('W5', shape=(3,3,128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_fc1': tf.get_variable('W6', shape=(28*28*128,1024), initializer=tf.contrib.layers.xavier_initializer()), 
    'W_fc2': tf.get_variable('W7', shape=(1024,1024), initializer=tf.contrib.layers.xavier_initializer()), 
    'Wout_gru1': tf.get_variable('W8', dtype = tf.float32,shape=(256,256), initializer=tf.contrib.layers.xavier_initializer()),
    'Wout_gru2': tf.get_variable('W9', dtype = tf.float32,shape=(512,vb_size), initializer=tf.contrib.layers.xavier_initializer())
    }
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc5': tf.get_variable('B4', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bc6': tf.get_variable('B5', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'b_fc1': tf.get_variable('B6', shape=(1024), initializer=tf.contrib.layers.xavier_initializer()),
    'b_fc2': tf.get_variable('B7', shape=(1024), initializer=tf.contrib.layers.xavier_initializer()),
    'Bout_gru1':tf.get_variable('B8', dtype = tf.float32,shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'Bout_gru2': tf.get_variable('B9', dtype = tf.float32,shape=(vb_size), initializer=tf.contrib.layers.xavier_initializer())
    }

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

class GRU:
    def __init__(self, input_dimensions, hidden_size, inputs,dtype=tf.float32):
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        self.input_layer=[]
        
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Wr = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wr')
        self.Wz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wz')
        self.Wh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wh')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Ur = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ur')
        self.Uz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uz')
        self.Uh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uh')
        
        # Biases for hidden vectors of shape (hidden_size,)
        self.br = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='br')
        self.bz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='bz')
        self.bh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='bh')
        
        # Define the input layer placeholder
        self.input_layer = inputs
        
        # Put the time-dimension upfront for the scan operator
        self.x_t = tf.transpose(self.input_layer, [1, 0, 2], name='x_t')
        
        # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
        self.h_0 = tf.matmul(self.x_t[0, :, :], tf.zeros(dtype=tf.float32, shape=(input_dimensions, hidden_size)), name='h_0')
        
        # Perform the scan operator
        self.h_t_transposed = tf.scan(self.forward_pass, self.x_t, initializer=self.h_0, name='h_t_transposed')
        
        # Transpose the result back
        self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')

    def forward_pass(self, h_tm1, x_t):
        
        # Definitions of z_t and r_t
        z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
        r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)
        
        # Definition of h~_t
        h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)
        
        # Compute the next hidden state
        h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

        return h_t

def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean, pop_var, beta, scale, epsilon)


def cnn_test(x,weights,biases):

    
  
    x = tf.reshape(x, shape=[-1, 224, 224, 3])
   
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1'])+  biases['bc1'])
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['bc2'])
   
    conv2 = maxpool2d(conv2)
 
    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['bc3'])

    conv4 = tf.nn.relu(conv2d(conv3, weights['W_conv4']) + biases['bc4'])

    
    conv4 = maxpool2d(conv4)

    
    conv5 = tf.nn.relu(conv2d(conv4, weights['W_conv5']) + biases['bc5'])
   
    conv6 = tf.nn.relu(conv2d(conv5, weights['W_conv6']) + biases['bc6'])
    
    
    conv6 = maxpool2d(conv6)
   
    fc1 = tf.reshape(conv6,[-1, weights['W_fc1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1'])+biases['b_fc1'])

    fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2'])+biases['b_fc2'])

    
    return fc2



def cnn_train(x,weights,biases):
   
    
    
  
    x = tf.reshape(x, shape=[-1, 224, 224, 3])
   
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1'])+  biases['bc1'])
   
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['bc2'])
  
    conv2 = maxpool2d(conv2)

    conv2 = tf.nn.dropout(conv2, 0.25)
 
    
    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['bc3'])
   
    conv4 = tf.nn.relu(conv2d(conv3, weights['W_conv4']) + biases['bc4'])

    
    conv4 = maxpool2d(conv4)
    
    conv4 = tf.nn.dropout(conv4, 0.25)
    
    conv5 = tf.nn.relu(conv2d(conv4, weights['W_conv5']) + biases['bc5'])
    
    conv6 = tf.nn.relu(conv2d(conv5, weights['W_conv6']) + biases['bc6'])
    
    conv6 = maxpool2d(conv6)
   
    conv6 = tf.nn.dropout(conv6, 0.25)

    fc1 = tf.reshape(conv6,[-1, weights['W_fc1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1'])+biases['b_fc1'])
   
    fc1 = tf.nn.dropout(fc1, 0.3)
    
    fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2'])+biases['b_fc2'])
    fc2 = tf.nn.dropout(fc2, 0.3)  

    return fc2
