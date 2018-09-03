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


![701](https://user-images.githubusercontent.com/20837727/44968671-2b474600-af17-11e8-8c3b-968cd91c0d9c.png)

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

Semantics-Preserving Locality Embedding for Zero-Shot Learning:

- This is the package with code and demo usage for the paper:</br>
- **Semantics-Preserving Locality Embedding for Zero-Shot Learning**</br>
- Author: Shih-Yen Tao, Yi-Ren Yeh, Yao-Hung Hubert Tsai and Yu-Chiang Frank Wang (*equal contribution)
- Published in British Machine Vision Conference (BMVC) 2017
- Prepare Data
	- Dowload data used in this paper at <https://drive.google.com/open?id=0B1QmFw8l-GM2V0ZyVXBMYUxrZ2M>
    - Put data in **/BMVC/Data**
    - Create **/BMVC/Param** and **/BMVC/result**
- Run Demo
	- Edit **/BMVC/Demo.m** with desired experiment. You are welcome to try different dataset or semantic vectors
	- Directly run **/BMVC/Demo.m** or **/BMVC/Demo_SUN.m**
	- The results will be saved in **/BMVC/result**
