import cv2
import sys
from os import listdir
from os.path import join
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

def resize_img(png_file_path):
        #image processing with resizing image for feeding into visual model of CNN
        img_rgb = cv2.imread(png_file_path)
        img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_adapted = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
        img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
        resized = cv2.resize(img_stacked, (224,224), interpolation=cv2.INTER_AREA)
        bg_img = 255 * np.ones(shape=(224,224,3))

        bg_img[0:224, 0:224,:] = resized
        bg_img /= 255
        bg_img = np.rollaxis(bg_img, 2, 0)  

        return bg_img
    
def load_doc(filename):
    file = open(filename, 'r',encoding='UTF-8')
    text = file.read()
    file.close()
    return text

class Dataset():
    def __init__(self, data_dir, input_transform=None, target_transform=None):
        self.data_dir = data_dir
        self.image_filenames = []
        self.texts = []
        all_filenames = listdir(data_dir)
        all_filenames.sort()
        for filename in (all_filenames):
            if filename[-3:] == "png":
                #only image files are stored into dateset.image
                self.image_filenames.append(filename)
            else:
                #adding START and END TAG for caption training
                text = '<START> ' + load_doc(self.data_dir+filename) + ' <END>'
                text = ' '.join(text.split())
                text = text.replace(',', ' ,')
                self.texts.append(text)
        self.input_transform = input_transform
        self.target_transform = target_transform
        
        # Initialize the function to create the vocabulary 
        tokenizer = Tokenizer(filters='', split=" ", lower=False)
        # Create the vocabulary 
        tokenizer.fit_on_texts([load_doc('vocabulary.vocab')])
        self.tokenizer = tokenizer
        # Add one spot for the empty word in the vocabulary 
        self.vocab_size = len(tokenizer.word_index) + 1
        # Map the input sentences into the vocabulary indexes
        self.train_sequences = tokenizer.texts_to_sequences(self.texts)
        # The longest set of boostrap tokens
        self.max_sequence = max(len(s) for s in self.train_sequences)
        # Specify how many tokens to have in each input sentence
        self.max_length = 48
        
        X, y, image_data_filenames = list(), list(), list()
        for img_no, seq in enumerate(self.train_sequences):
             #input_sequence prepends start sequence
             #output_sequence appends end sequence
            in_seq, out_seq = seq[:-1], seq[1:]
              #changing outsequence to one hot encoding 
            out_seq = to_categorical(out_seq, num_classes=self.vocab_size)
            image_data_filenames.append(self.image_filenames[img_no])
                #dataset.X holds the insequence
            X.append(in_seq)
                #dataset.Y holds the outsequence
            y.append(out_seq)
                
        self.X = X
        self.y = y
        self.image_data_filenames = image_data_filenames
        self.images = list()
        for image_name in self.image_data_filenames:
            image = resize_img(self.data_dir+image_name)
            self.images.append(image)
import functools 
def pad(batch_y):
        #adding padding to the sequences for keeping uniform sequence length across batches
    print(batch_y.shape)
    x=0
    for y in batch_y:
        if(len(y)>x):
            x=len(y)

    
    ret = []
    for y in range(len(batch_y)):
        res=np.zeros(x)
        s = batch_y[y]
        res[0:len(s)]=batch_y[y]

        ret.append(res)
    return np.array(ret)
        
        

def pad2(batch_ex):

    r=0
    c=0
    for ex in batch_ex:
        shape = ex.shape

        if(shape[0]>r):
            r=shape[0]
        if(shape[1]>c):
            c=shape[1]

    ret=[]
    for ex in batch_ex:
        res=np.zeros((r,c))

        res[0:ex.shape[0],0:ex.shape[1]]=ex
        ret.append(res)

        
    return(np.array(ret))

def word_for_id(integer, tokenizer):
        #for each word, get its corresponding tokenized id
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
def load_val_images(data_dir):
        #for testing images. do the image processing
    image_filenames =[]
    images = []
    all_filenames = listdir(data_dir)
    all_filenames.sort()
    for filename in (all_filenames):
        if filename[-3:] == "png":
            image_filenames.append(filename)
    for name in image_filenames:
        image = resize_img(data_dir+name)
        images.append(image)
    return images
