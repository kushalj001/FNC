import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout

load_df = pd.read_pickle('./updated_stance')
train_X = np.array(load_df['input'])
labels = np.array(load_df['Stance'])

#print(train_X.shape)

X = np.array(load_df['input'].values.tolist(),dtype=np.float32)

#print(X)

one_hot = pd.get_dummies(load_df['Stance'])
#print(one_hot.shape)
pickle.dump(one_hot,open('one_hot_labels.pk','wb'))
train_X = X


#  TensorFlow Model
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def model(X,weights_hidden,weights_output):
    hidden = tf.nn.relu(tf.matmul(X,weights_hidden))
    return tf.matmul(hidden,weights_output)

X = tf.placeholder(dtype=tf.float32,shape=[None,10001],name='X')
Y = tf.placeholder(dtype=tf.float32,shape=[None,4],name='Y')

weights_hidden = init_weights([10001,100])
weights_output = init_weights([100,4])

model_output = model(X,weights_hidden,weights_output)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_output,labels=Y))
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
predict_op = tf.argmax(model_output,1)

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10):
        print(i)
        sess.run(train_op,feed_dict={X:train_X,Y:one_hot})


    save_path = saver.save(sess,'./model.ckpt')
    print('Model saved in: ', save_path)
    matches = tf.equal(tf.argmax(model_output,1),tf.argmax(one_hot,1))
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    print(sess.run(acc,feed_dict={X:train_X,Y:one_hot}))




# Keras model

model = Sequential()
model.add(Dense(units=100,activation='relu',input_shape=(10001,),name='dense_1'))
model.add(Dropout(0.6,name='dropout_1'))
# model.add(Dense(units=100,activation='relu',name='dense_2'))
# model.add(Dropout(0.6,name='dropout_2'))
model.add(Dense(units=4,activation='softmax',name='output'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X,labels,epochs=3)
model.save('keras_model_updated')
