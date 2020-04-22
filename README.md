# Automated HTML code generation using Deep learning (tensorflow version 1.14)

## Abstract
This project aims on cutting down development time of web UI design by generating responsive web layouts straight from images of webpages or even hand-drawn sketches of webpages that is, to generate HTML code fromimages.Thisproblemcanbemodelledasanimagecaptioning problem where the output language consists of a series of tokens which can be converted into HTML code with the help of a compiler.

## Dataset
The dataset includes 1750 images(screenshots) and 1750 corresponding DSL(corresponding code). The training data involves 1500 images and DSLs and testing, 250 images and DSLs. The images in the dataset are of the dimension 2400 x 1380 pixels. The DSL has a vocabulary of a total 18 words.
</br>
https://github.com/tonybeltramelli/pix2code/tree/master/datasets

## Model Architecture
</br>
#### Overall model
</br>
<img src="/images/overall.png" width="50%" height="50%"/>
</br>
<table>
 <tr>
  <td>Visual model in CNN</td>
  <td>language model for encoder</td>
 <tr>
 <tr>
  <td rowspan=3><img src="/images/cnn.PNG" width="75%" height="75%"/></td>
  <td><img src="/images/encoder.PNG" width="75%" height="75%"/></td>
 </tr>
 <tr>
  <td>Decoder</td>
 </tr>
 <tr>
  <td><img src="/images/decoder.PNG" width="75%" height="75%"/></td>
 </tr>
</table>

## Implementation
### Preprocessing the Dataset 
The image is resized to 3x224x224 px image. Further the DSL is processed as 2 new components, namely the in-sequence and the out-sequence. The in-sequence consists of the caption prepended with a <START >and the out-sequence is the caption appended with the <END > tag.The is further tokenized and changed into its respective one hot encoding.
 
### Encoder CNN
The CNN employed to do feature engineering on given images and return a feature map to the language model to work on. Finally the output feature matrix is repeated ’N’ number of times where N represent the number of words in the caption sequence corresponding to that image. 

### Encoder RNN 
Encoder RNN has been implemented using a Gated Recurrent Unit(GRU). The DSL is ﬁrst fed into an embedding layer which is then fed into the two layers of GRU 

### Decoder RNN 
The outputs on Encoder CNN (feature vector) and Encoder RNN  are concatenated in the third dimension and fed as input to the Decoder RNN. The output of this decoder is ﬁnally the set of tokens which correspond to the DSL.

### Predicting tokens for new images 
For the ﬁnal testing, the new sample image is fed into the CNN and corresponding to this the <START >tag is given as input to the encoder RNN. These inputs produce text sequences till the <END >tag is encountered. Further the generated DSL is compared with original DSL and a BLEU score is calculated.
 
### Running the Compiler
The generated DSL is fed into the Compiler which compares each token to a JSON where each word in the vocabulary is mapped to its respective HTML format and produces a syntactically accurate HTML code

## Files and Folders in the Repository
Outputs : trial version</br>
all trials : notebooks for experimenting with each model of our code</br>
images : images of model architecture</br>
sampledata_predict : single image and gui for testing</br>
Compiler.py - Convert DSL to HTML code</br>
Dataset.py - Functions for modifying and preprocessing data</br>
Embedding.ipynb - Word2Vec implemented in tensorflow</br>
Integrated.ipynb: Entire code in a single notebook</br>
LSTM_comparison.ipynb - Comparing performance of LSTM and GRU</br>
Main.py - testing and prediction code</br>
Model_Utils.py - tensor flow implementation of GRU and CNN</br>
train.py - code for training</br>
try.zip : subset of dataset</br>
Presentaion.pptx : Formal presentation</br>
Report.pdf: Formal IEEE formatted Report</br>
vocabulary.vocab : 18 words used in our vocabulary</br>

## Results
</br>
<table>
 <tr>
  <td>Input Screenshot of webpage</td>
  <td>Output HTML rendered by our model</td>
 </tr>
 <tr>
  <td><img src="/images/try_done.PNG"/></td>
  <td><img src="/images/test_done.PNG"/></td>
 </tr>
 </table>


## Contributors
#### Namrata R
#### Pragnya Sridhar
#### Sarang Ravindra
