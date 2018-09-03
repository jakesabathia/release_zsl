Machine Learning for Zero-Shot classification
=========================
Author: Shih-Yen Tao <shihyent@andrew.cmu.edu> </br>
=========================

This page includes my approaches toward Zero-Shot classification, that is to recognize classes which are not
presented in the training stage. In particular, I have come up with one Deep Learning method and one traditional machine learning algorithm.</br>
1. **Predict Deep Latent Classifier for Unseen Category** </br>
2. **Semantics-Preserving Locality Embedding for Zero-Shot Learning** </br>

Predict Deep Latent Classifier for Unseen Category
------

![Image description](https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwjq1pOXg57dAhXxg-AKHb7GB5EQjRx6BAgBEAU&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DSfLV8hD7zX4&psig=AOvVaw0LS1OJEc1Bv5tw1DH43fbt&ust=1536036276690998)

- This is the project for **CMU 10701 (PhD level Introduction to Machine Learning)**. I designed a deep learning algorithm which can predict the visual classifiers for unseen categories.
- Prepare Data
	- Download data at <https://drive.google.com/drive/folders/1Fqs7uBFI-BlRrcQXxIDvs0hCpfFoxXew?usp=sharing>
	- Put data under **/DEEP/Data/**
- Install Tensorflow
	- Please follow the instruction on <https://www.tensorflow.org/> to install Tensorflow.
- Run demo code
	- By directly running **/DEEP/Demo.py** should start training the network
	- You can choose the dataset to work on in the code (AWA, CUB, DOG). Moreover, you are welcome to play with different semantic vectors (attribute, word2vec, glove, wordnet).
- For implementation details please refer to my report <https://github.com/jakesabathia/jakesabathia.github.io/blob/master/paper/10701.pdf>


#####Package with code and demo usage for the paper:</br>
#####"Semantics-Preserving Locality Embedding for Zero-Shot Learning"</br>
#####    Shih-Yen Tao, Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang</br>
#####    British Machine Vision Conference (BMVC) 2017.

Setup:
------
- Prepare Data
	- Dowload data used in this paper at <http://jakesabathia.github.io>
    - Put data in **/Data**
    - Create */Param* and */result*

- Edit **/code/Demo.m** with desired experiment
    - Try different dataset or semantic vector if you want

Run:
-----
- Directly run **/code/Demo.m** or **/code/Demo_SUN.m**
- The results will be saved in */result*
