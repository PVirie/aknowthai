# **AKnowThai**

Thai OCR with deep learning
  - For neural network crazes
  - End-to-end training 
  		- only need images of characters and a list of characters in unicode
  		- no need to provide stop marks or character cuts
  - with Accuracy

## Install prerequisites

1. Python 2.7.x 
2. Cuda Toolkit 8.0 (https://developer.nvidia.com/cuda-toolkit)
3. TensorFlow r1.0 (https://www.tensorflow.org/install/)


## Project structure

1. Run python src/train.py to train.
2. Run python src/test.py to test.


## New fonts

1. Put new fonts under the "fonts" directory.
	- If fonts have different spacing diversity, the model will be hard to converge. (Need to go deeper?)

## More

https://pvirie.wordpress.com/2016/11/23/a-challenge-to-test-neural-generative-attention/

## License 

MIT

**That means free software.**


