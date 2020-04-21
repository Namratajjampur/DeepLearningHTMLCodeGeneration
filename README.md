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
![Screenshot](/images/overall.png)
</br>
Visual model in CNN
</br>
![header image](https://raw.github.com/Namratajjampur/DeepLearningHTMLCodeGeneration/tree/master/images/cnn.PNG)
</br>
language model for encoder
</br>
![Image description](https://github.com/Namratajjampur/DeepLearningHTMLCodeGeneration/tree/master/images/encoder.PNG)
</br>
Decoder 
</br>
![Screenshot](https://github.com/Namratajjampur/DeepLearningHTMLCodeGeneration/tree/master/images/decoder.PNG)

## Implementation

## Files in the Repository

## Results
</br>
Input Screenshot of webpage
</br>
![Screenshot](https://github.com/Namratajjampur/DeepLearningHTMLCodeGeneration/tree/master/images/try_done.PNG)
</br>
Ouput HTML rendered by our model
</br>
![Screenshot](https://github.com/Namratajjampur/DeepLearningHTMLCodeGeneration/tree/master/images/test_done.PNG)

##Contributors
Namrata R
Pragnya Sridhar
Sarang Ravindra
