import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

categories = ["Fname","Lname","Designation","Organisation","Pincode","State","City","country","Mobile", "Emails", "Website"]
words = []
featurev = []
docs = [[]]

#creating bag of words
def make_dict(dataset):
    size = len(dataset)
    for m in range(size):
        mail = dataset["data"][m]
        mail = list(str(mail).strip('\n'))
        docs.append((mail,dataset["classes"][m]))
    words = ['@','.','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','9','8','7','6','5','4','3','2','1','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']    
    return(words,docs)

features = []
#creating feature vector and output vector
def testdata(words,docs):
    print('starting feature extraction')
    for doc in docs:
        if(doc):
            bow = []
            for w in words:
                bow.append(doc[0].count(w))                    
            output_row = categories.index(doc[1])
            bow.append(output_row)          
            features.append(bow)
    return(features)
    

datas = pd.read_csv('/home/anjana/Desktop/train_set.csv', encoding='latin-1')
dataset = datas.loc[:, ~datas.columns.str.contains('^Unnamed')]
words,docs = make_dict(dataset)
features = testdata(words,docs)
print(len(features))

def read_dataset(features):
    features = np.asarray(features)
    
    X = features[:,:-1]
    y = features[:,-1:]
    #X = df[df.columns[0:63]].values
    #y = df[df.columns[63]]
    
    #encode the independent variables
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    #print(X.shape)
    return(X,Y)
    

#define encoder function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    print(n_unique_labels)
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return(one_hot_encode)
    

X, Y = read_dataset(features)
#print(X[0])
#print(len(Y[0]))

#shuffle dataset and mix up rows
X, Y=shuffle(X,Y,random_state = 1)

#convert dataset into train and test part
train_x,test_x,train_y,test_y=train_test_split(X,Y, test_size = 0.2)

#inspect shape of training and testing data
print("train_x shape : ",train_x.shape)
print("train_y shape : ",train_y.shape)
print("test shape : ",test_x.shape)
print(train_y[0])


#define the important parameters and variables to work with tensors
learning_rate = 0.3
training_epochs = 100
cost_history = np.empty(shape=[1], dtype = float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = len(categories)
model_path = "/home/anjana/Desktop/OCR"

#define number of hidden layers
n_hidden1 = 10
n_hidden2 = 10

x = tf.placeholder(tf.float32,[None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])
print(y_.shape)

#define the model
def multilayer_perceptron(x, weights, biases):
    #hidden layer with relu activised
    layer1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer1 = tf.nn.sigmoid(layer1)
    
    #hidden layer with sigmoid activation
    layer2 = tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
    layer2 = tf.nn.sigmoid(layer2)
    
    #output layer
    out_layer = tf.matmul(layer2, weights['out']) + biases['out']
    return (out_layer)

weights = {
        'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden1])),
        'h2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2])),
        'out': tf.Variable(tf.truncated_normal([n_hidden2, n_class])),
        }

biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden1])),
        'b2': tf.Variable(tf.truncated_normal([n_hidden2])),
        'out': tf.Variable(tf.truncated_normal([n_class])),
        }

init = tf.global_variables_initializer()

saver = tf.train.Saver()
y = multilayer_perceptron(x, weights, biases)
#define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))

training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
sess = tf.Session()
sess.run(init)

#calculate the cost and accuracy for each epoch
mse_history = []
accuracy_history = []
for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history,cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy: ", (sess.run(accuracy,feed_dict={x:test_x, y_:test_y})))
    pred_y = sess.run(y, feed_dict = {x:test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x:train_x, y_:train_y}))
    accuracy_history.append(accuracy)
    print('epoch: ', epoch, '-', 'cost: ', cost, " -MSE: ", mse_, "-Train Accuracy: ", accuracy)

save_path = saver.save(sess, model_path)
print("Model saved in file: %s" %save_path)


#plot mse and accuracy graph

plt.plot(mse_history, 'r')
plt.show()
plt.plot(accuracy_history)
plt.show()

#print the final accuracy

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_:test_y})))

#print the final mean square error

pred_y = sess.run(y, feed_dict={x:test_x}) 
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))

