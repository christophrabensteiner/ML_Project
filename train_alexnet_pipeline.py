import tensorflow as tf
import os
import alexnet
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import data_alex

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

training_dir = '/home/valentin/Desktop/MLProject/data/fashion_png/training'
testing_dir = '/home/valentin/Desktop/MLProject/data/fashion_png/testing'

imageWidth = 28
imageHeight = 28
imageSize = 28 * 28
NChannels = 1
NClasses = 10
BATCH_SIZE = 128

EPOCHS_List = [10,50,100,150]
rate_List = [0.01,0.001]
dropout_List = [1.0,0.75,0.5,0.25] # Dropout, probability to keep units

#Load Data through pipeline
X_train, y_train = data_alex.LoadTrainingData(training_dir, (imageWidth, imageHeight), float(0))
data_alex.TrainingData = X_train
data_alex.TrainingLables = y_train

X_test, y_test, NamesT, _, Paths = data_alex.LoadTestingData(testing_dir, (imageWidth, imageHeight))
data_alex.TestingData = X_test
data_alex.TestingLables = y_test

# Training / Validation Split
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.08333, random_state=1)


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
            log_dir = '/home/valentin/Desktop/MLProject/logs/' + MODEL + '/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            #placeholders for model
            x = tf.placeholder(tf.float32, (None, 28, 28, 1))
            y = tf.placeholder(tf.int32, (None))
            one_hot_y = tf.one_hot(y, 10)
            keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

            # load AlexNet
            logits = alexnet.alex_net(x, keep_prob)


            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
            loss_operation = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=rate)
            training_operation = optimizer.minimize(loss_operation)

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
            accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            train_prediction = tf.nn.softmax(logits)
            saver = tf.train.Saver()

            #Tensorboard
            tf.summary.histogram('cross_entropy', cross_entropy)
            tf.summary.histogram('predictions', train_prediction)
            tf.summary.scalar('loss_operation', loss_operation)
            tf.summary.scalar('accuracy_operation', accuracy_operation)

            # evaluate model accuracy with validation set
            def evaluate(X_data, y_data):
                num_examples = len(X_data)
                total_accuracy = 0
                sess = tf.get_default_session()
                for offset in range(0, num_examples, BATCH_SIZE):
                    batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
                    accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                    total_accuracy += (accuracy * len(batch_x))
                return total_accuracy / num_examples


            # TRAINING of MODEL
            with session as sess:

                train_writer = tf.summary.FileWriter(log_dir, sess.graph)

                sess.run(tf.global_variables_initializer())
                num_examples = len(X_train)
                rang = num_examples / BATCH_SIZE

                print("Training...")
                print()

                total_Iterations = 0

                for i in range(EPOCHS):
                    avg_loss = 0.0
                    avg_acc = 0.0
                    X_train, y_train = shuffle(X_train, y_train)
                    merge = tf.summary.merge_all()
                    for offset in range(0, num_examples, BATCH_SIZE):
                        total_Iterations += 1
                        end = offset + BATCH_SIZE
                        batch_x, batch_y = X_train[offset:end], y_train[offset:end]


                        _, loss, predictions, summary= sess.run([training_operation, loss_operation, train_prediction, merge],
                                                        feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

                        avg_loss += (loss / (rang))
                        if (end % (BATCH_SIZE*10)) == 0:
                            merge = tf.summary.merge_all()
                            train_writer.add_summary(summary, total_Iterations)
                            validation_accuracy = evaluate(X_validation, y_validation)
                            print("EPOCH {} ...".format(i + 1))
                            print("Iteration = {:.0f} Batches = {:.0f} Loss= {:.5f} Validation Accuracy = {:.3f}" .format((end/BATCH_SIZE), end, loss, validation_accuracy))
                            print()


                saver.save(sess, model_dir+MODEL)
                print("Model saved")

            # TESTIN of MODEL
            with tf.Session() as sess:
                print(model_dir+MODEL)
                saver.restore(sess, tf.train.latest_checkpoint(model_dir))
                test_accuracy = evaluate(X_test, y_test)
            print("Test Accuracy = {:.3f}".format(test_accuracy))
            tf.reset_default_graph()
