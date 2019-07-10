import tensorflow as tf
import os
import alexnet
import data_alex
from sklearn.metrics import confusion_matrix
import csv
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

testing_dir = '/home/valentin/Desktop/MLProject/data/fashion_png/testing'

imageWidth = 28
imageHeight = 28
NClasses = 10
BATCH_SIZE = 128

dropout_List = [1.0,0.75,0.5,0.25] # Dropout, probability to keep units
EPOCHS_List = [10,50,100]
rate_List = [0.01,0.001]

#Load Data through pipeline
X_test, y_test, NamesT, _, Paths = data_alex.LoadTestingData(testing_dir, (imageWidth, imageHeight))
data_alex.TestingData = X_test
data_alex.TestingLables = y_test


for EPOCHS in EPOCHS_List:
    for dropout in dropout_List:
        for rate in rate_List:

            #GPU config
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)

            #path conifg
            MODEL = 'alexnet_E' +str(EPOCHS)+ '_D' + str(dropout) + '_R' + str(rate)
            model_dir = '/home/valentin/Desktop/MLProject/models/'+MODEL+'/'
            save_dir = '/home/valentin/Desktop/MLProject/results/'+MODEL+'/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            #placeholders for model
            x = tf.placeholder(tf.float32, (None, 28, 28, 1))
            y = tf.placeholder(tf.int32, (None))
            one_hot_y = tf.one_hot(y, 10)
            keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

            # load LeNet
            logits = alexnet.alex_net(x, keep_prob)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
            loss_operation = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=rate)
            training_operation = optimizer.minimize(loss_operation)

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
            accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            train_prediction = tf.nn.softmax(logits)
            output = tf.argmax(logits,1)

            # evaluate model accuracy with validation set
            def evaluate(X_data, y_data):
                sess = tf.get_default_session()
                accuracy, predictions = sess.run([accuracy_operation,output],
                                    feed_dict={x: X_test, y: y_test, keep_prob: 1.})
                matrix = confusion_matrix(y_test, predictions)
                return accuracy, matrix

            # TESTIN of MODEL
            with tf.Session() as sess:
                print(model_dir+MODEL)
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(model_dir))
                test_accuracy, conf_matrix = evaluate(X_test, y_test)
                print(conf_matrix)

            print("Test Accuracy = {:.3f}".format(test_accuracy))

            #write results to csv
            with open(save_dir+"Results_"+MODEL+".csv", 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(list(conf_matrix))
                writer.writerow("")
                writer.writerow(np.asarray([test_accuracy]))
            tf.reset_default_graph()

