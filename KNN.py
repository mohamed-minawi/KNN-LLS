import numpy as np

class KNN(object):
    def __init__(self):
        pass

    def train(self, Data, Label):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.train_data = Data
        self.train_labels = Label

    def predict(self, test_data, l='L1', k=1):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = test_data.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.train_labels.dtype)
        # loop over all test rows
        for i in range(num_test):
            # find the nearest training example to the i'th test example
            if l == 'L1':
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.train_data - test_data[i, :]), axis=1)
            else:
                distances = np.sqrt(np.sum(np.square(self.train_data - test_data[i, :]),axis=1))

            closelabels = self.train_labels[distances.argsort()[:k]]
            count = np.bincount(closelabels);
            Ypred[i] = np.argmax(count)
            print(k,": Test ", i," done ")
        return Ypred