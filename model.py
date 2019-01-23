import tensorflow as tf
import keras
from LDNet import DefenseLayer


class Model:
	def __init__(self,ldnet=False,height=28,width=28,channels=1,classes=10):
		self.height=height
		self.width=width
		self.channels=channels
		self.image_size=height
		self.num_labels=classes
		self.num_channels=channels
		x=keras.layers.Input(shape=(self.height,self.width,self.channels))
		if(ldnet):
			temp=DefenseLayer()(x)
			newx=keras.layers.Add()([x,temp])
			c1=keras.layers.Conv2D(filters=32,kernel_size=5,activation='relu')(newx)
		else:
			c1=keras.layers.Conv2D(filters=32,kernel_size=5,activation='relu')(x)

		m1=keras.layers.MaxPooling2D(pool_size=(2,2))(c1)
		m1=keras.layers.Dropout(0.2)(m1)

		f=keras.layers.Flatten()(m1)
		d1=keras.layers.Dense(128, activation='relu')(f)
		d2=keras.layers.Dense(50,activation='relu')(d1)
		logits=keras.layers.Dense(classes)(d2)

		self.model=keras.models.Model(inputs=x,outputs=logits)

	def train(self, X, Y, valid_x=None, valid_y=None, epochs=10, batch_size=100):
		def loss(actual,predicted):
			return tf.nn.softmax_cross_entropy_with_logits(logits=predicted, labels=actual)

		self.model.compile(loss=loss, optimizer='adam',  metrics=['acc'])
		if(valid_x is None):
			self.model.fit(X,Y,batch_size=batch_size,epochs=epochs)
		else:
			self.model.fit(X,Y,validation_data=(valid_x,valid_y),batch_size=batch_size,epochs=epochs)

	def predict(self,x):
		return self.model(x)