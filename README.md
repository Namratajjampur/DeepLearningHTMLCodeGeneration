# Automated HTML code generation using Deep learning (tensorflow version 1.14)

## Abstract
This project aims on cutting down development time of web UI design by generating responsive web layouts straight from images of webpages or even hand-drawn sketches of webpages that is, to generate HTML code fromimages.Thisproblemcanbemodelledasanimagecaptioning problem where the output language consists of a series of tokens which can be converted into HTML code with the help of a compiler.

## Dataset
The dataset includes 1750 images(screenshots) and 1750 corresponding DSL(corresponding code). The training data involves 1500 images and DSLs and testing, 250 images and DSLs. The images in the dataset are of the dimension 2400 x 1380 pixels. The DSL has a vocabulary of a total 18 words.
</br>
https://github.com/tonybeltramelli/pix2code/tree/master/datasets

## Model Architecture
</br>
Overall model
</br>
<img src="/images/overall.png" width="25%" height="25%"/>
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
Namrata R</br>
Pragnya Sridhar</br>
Sarang Ravindra
