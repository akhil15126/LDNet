import tensorflow as tf
import keras
import numpy as np
from keras.datasets import mnist
from l2_attack import CarliniL2
from model import Model
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ldn", "--ldnet", action='store_true', help="Use Defense Layer")
parser.add_argument("-e","--epochs", type=int, default=1, help="Number of epochs")
args = parser.parse_args()


with tf.Session() as sess:
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train=X_train.reshape((X_train.shape[0],28,28,1))
	X_test=X_test.reshape((X_test.shape[0],28,28,1))

	num_classes=10
	Y_train = keras.utils.to_categorical(Y_train, num_classes)
	Y_test = keras.utils.to_categorical(Y_test, num_classes)

	X_train=X_train/255.0
	X_test=X_test/255.0

	model=Model(ldnet=args.ldnet)

	model.train(X_train,Y_train,epochs=args.epochs)

	attack=CarliniL2(sess,model,batch_size=1,targeted=False,boxmin=0.0,boxmax=1.0)
	adv=attack.attack(X_test,Y_test)