import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import CNN
import random
# import cnn_utils


def random_mini_batches(train_x, train_y, minibatch_size, seed):
    minibatches=[]
    random.seed(seed)
    choose = random.sample(range(0,train_x.shape[0]-1),minibatch_size*5)
    for k in range(5):
        minibatch_x = []
        minibatch_y = []
        # print(choose)
        for i in range(minibatch_size):
            minibatch_x.append(train_x[choose[k*minibatch_size+i]])
            minibatch_y.append(train_y[choose[k*minibatch_size+i]])
        minibatches.append((minibatch_x,minibatch_y))

    return minibatches

def model(train_x, train_y, test_x, test_y, lr=0.01, num_epoch=100, minibatch_size=64, print_cost=True, isPlot=True, train=True):

    # seed = 3
    # costs = []
    # (m, n_H0, n_W0, n_C0) = train_x.shape
    # n_y = train_y.shape[1]
    # parameters = CNN.initialize_parameters()

    model = CNN.build()
    print("Label size:", train_y.shape)
    model.fit(train_x, train_y, batch_size = minibatch_size, epochs = num_epoch)
    print("-----------------测试集评估-----------------")
    loss, accuracy = model.evaluate(test_x,test_y)
    print("loss: ", loss, "准确率: ", accuracy)
    model.summary()
    model.save("mnist-dataset/model")
    
    # X, Y = CNN.create_placeholder(minibatch_size, n_H0, n_W0, n_C0, n_y)
    # Z4 = CNN.build(X,parameters)
    # cost = CNN.compute_cost(Y, Z4)
    # # Z4 = tf.reshape(Z4,[minibatch_size,25])
    # # Y = tf.reshape(Y,[minibatch_size,25])
    # # Z4_ = tf.Variable(Z4)
    # # Y_ = tf.Variable(Y)
    # # loss = lambda:tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(Y_,Z4_))
    # optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(cost)
    # init = tf.compat.v1.global_variables_initializer()

    # # parameters = CNN.initialize_parameters()
    # # Z4.set_shape([minibatch_size,Z4.shape[1]])
    # # Y.set_shape([minibatch_size,Y.shape[1]])
    
    # # model_path = "mnist-dataset/img-model"
    # # saver = tf.keras.Model.save_weight(model_path)
    # with tf.compat.v1.Session() as session:
    #     if train:
    #         session.run(init)
    #         for epoch in range(num_epoch):
    #             epoch_cost = 0
    #             num_minibatches = int(m / minibatch_size)
    #             seed = seed + 1
    #             minibatches = random_mini_batches(train_x, train_y, minibatch_size, seed)
    #             for minibatch in minibatches:
    #                 (minibatch_x, minibatch_y) = minibatch
    #                 # minibatch_x = tf.reshape(minibatch_x,[minibatch_size,28,28,1])
    #                 # minibatch_y = tf.reshape(minibatch_y,[minibatch_size,25])
    #                 result, minibatch_cost = session.run([Z4,cost,optimizer], feed_dict={X:minibatch_x, Y:minibatch_y})[0:2]
    #                 epoch_cost += minibatch_cost / num_minibatches
    #             print(epoch_cost)
    #             costs.append(epoch_cost)
    #             if print_cost:
    #                 if epoch % 10 == 0:
    #                     print("epoch =", epoch, "epoch_cost =", epoch_cost)
    #         if isPlot:
    #             plt.plot(np.squeeze(costs))
    #             plt.title("learning_rate =" + str(lr))
    #             plt.xlabel("epoch")
    #             plt.ylabel("cost")
    #             plt.show()
    #         # parameters = session.run(parameters)
    #         correct_prediction = tf.equal(tf.argmax(Z4, axis=1), tf.argmax(Y, axis=1))
    #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #         print("训练集准确率：", accuracy.eval({X:train_x, Y:train_y}))
        # print("测试集准确率：", accuracy.eval({X:test_x, Y:test_y}))
    
        # else:
        #     # saver.restore(session,model_path)
        #     print("testing")
        #     model = tf.keras.models.load_model("mnist-dataset/model")
        #     model.fit(train_x,train_y)
        #     correct_prediction = tf.equal(tf.argmax(Z4, axis=1), tf.argmax(Y, axis=1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #     print("训练集准确率：", accuracy.eval({X:train_x, Y:train_y}))
        #     # minibatches = random_mini_batches(train_x, train_y, minibatch_size, seed)
        #     # for minibatch in minibatches:
        #     #     (minibatch_x, minibatch_y) = minibatch
        #     #     result, minibatch_cost = session.run([parameters,Z4,cost,optimizer], feed_dict={X:minibatch_x, Y:minibatch_y})[1:3]
        #     # correct_prediction = tf.equal(tf.argmax(Z4, axis=1), tf.argmax(Y, axis=1))
        #     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #     # print("训练集准确率：", accuracy.eval({X:train_x, Y:train_y}))
        #     return parameters