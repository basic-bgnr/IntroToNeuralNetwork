#!/bin/bash
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --output ./mnist_dataset/train-images-idx3-ubyte.gz 
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --output ./mnist_dataset/train-labels-idx1-ubyte.gz 
  
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  --output ./mnist_dataset/t10k-images-idx3-ubyte.gz  
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  --output ./mnist_dataset/t10k-labels-idx1-ubyte.gz  
