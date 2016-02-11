import theano
import theano.tensor as T

import numpy as np
from sklearn import datasets

import sklearn

class TheanoNN(object):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, reg_lambda, train_x, train_y):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        self.examples_count = train_x.shape[0]
        self.train_x = train_x
        self.train_y = train_y

        self.W1 = theano.shared(np.random.randn(input_dim,hidden_dim).astype('float32'), name='W1')
        self.b1 = theano.shared(np.zeros(hidden_dim).astype('float32'), name='b1')

        self.W2 = theano.shared(np.random.randn(hidden_dim,output_dim).astype('float32'), name='W2')
        self.b2 = theano.shared(np.zeros(output_dim).astype('float32'), name='b2')

        self.X = theano.shared(train_x.astype('float32'))
        self.Y = theano.shared(train_y.astype('float32'))

        z1 = self.X.dot(self.W1) + self.b1
        a1 = T.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        y_hat = T.nnet.softmax(z2)

        loss_reg = 1. / self.examples_count * self.reg_lambda/2 * (T.sum(T.sqr(self.W1)) + T.sum(T.sqr(self.W2)))
        loss = T.nnet.categorical_crossentropy(y_hat, self.Y).mean() + loss_reg
        prediction = T.argmax(y_hat, axis=1)

        dW2 = T.grad(loss, self.W2)
        db2 = T.grad(loss, self.b2)
        dW1 = T.grad(loss, self.W1)
        db1 = T.grad(loss, self.b1)



        self.feed_forward = theano.function([], y_hat)
        self.calculate_loss = theano.function([] , loss)
        self.predict = theano.function([], prediction)

        self.learning_step = theano.function([], updates=( (self.W2, self.W2 - self.learning_rate * dW2),
                                                               (self.W1, self.W1 - self.learning_rate * dW1),
                                                               (self.b2, self.b2 - self.learning_rate * db2),
                                                               (self.b1, self.b1 - self.learning_rate * db1)))

    def initialize_parameters(self):
        np.random.seed(0)
        self.W1.set_value(np.random.randn(self.input_dim, self.hidden_dim).astype('float32') / np.sqrt(self.input_dim))
        self.b1.set_value(np.zeros(self.hidden_dim).astype('float32'))
        self.W2.set_value(np.random.randn(self.hidden_dim, self.output_dim).astype('float32') / np.sqrt(self.hidden_dim))
        self.b2.set_value(np.zeros(self.output_dim).astype('float32'))

    def train(self, nepoch=20000, print_loss=True):
        self.initialize_parameters()

        for i in range(0, nepoch):
            self.learning_step()

            if print_loss and i % 10 == 0:
                print("Loss after iteration %i: %f" %(i , self.calculate_loss()))

    @staticmethod
    def test():
        np.random.seed(0)

        train_x, train_y = datasets.make_moons(5000, noise=.20)
        train_y = np.eye(2)[train_y]

        example_count = len(train_x)

        nn = TheanoNN(train_x.shape[1],1000,train_y.shape[1],np.float32(0.01),np.float32(0.01),train_x,train_y)
        nn.train()






if __name__ == '__main__':
    TheanoNN.test()
