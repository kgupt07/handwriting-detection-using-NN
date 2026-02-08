import numpy as np

#lets define prameters!!

#defining sigmoid: smoothening activation function to make our function non-linear
def sigmoid(x):
  x = np.clip(x, -500, 500)  # to prevent overflow 
  return 1/(1+np.exp(-x))

#defining its derivative:
def sigmoid_derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))

#defining loss function: mean square error with a factor of 0.5
def loss_function(Y, Y_hat):
  return 0.5 * np.sum(np.square(Y-Y_hat))/Y.shape[0]
  # the 0.5 exists there to a. cleaner math: derivative wont have factor
  # and b. it also works as a scaling for the learning rate.

#defining derivative of loss function wrt Y_hat
def loss_function_derivative(Y, Y_hat):
  return (Y_hat-Y)/Y.shape[0]

#lets turn the label to 'one-hot-ecoding' basically make the final fully
# connected layer of our ffn

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        #initializing layer sizes
        self.input_size = input_size          #784 = 28*28
        self.hidden_size = hidden_size        #16
        self.output_size = output_size        #10 coz we have 10 classes of nums
        #initializing our 2 weight matrices
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        #initializing the bias values
        self.b1 = np.zeros(self.hidden_size)
        self.b2 = np.zeros(self.output_size)

    def forward_pass(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward_pass(self, X, Y, learning_rate):
      loss = loss_function(Y, self.a2)   #loss is a number

      dL_da2 = loss_function_derivative(Y, self.a2)  #this is an array
      da2_dz2 = sigmoid_derivative(self.z2)
      dL_dz2 = dL_da2 * da2_dz2
      dz2_dw2 = self.a1

      # calcuate the gradient of the loss with respect to w2
      dL_dw2 = np.dot(dz2_dw2.T, dL_dz2)
      # calculate the gradient of the loss with respect to b2
      dL_db2 = np.sum(dL_dz2, axis=0)

      dz2_da1 = self.w2
      da1_dz1 = sigmoid_derivative(self.z1)
      dL_dz1 = np.dot(dL_dz2, dz2_da1.T) * da1_dz1
      dz1_dw1 = X

      # calculate the gradient of the loss with respect to w1
      dL_dw1 = np.dot(dz1_dw1.T, dL_dz1)
      # calculate the gradient of the loss with respect to b1
      dL_db1 = np.sum(dL_dz1, axis=0)

      # update the weights and biases
      self.w2 -= learning_rate * dL_dw2
      self.b2 -= learning_rate * dL_db2
      self.w1 -= learning_rate * dL_dw1
      self.b1 -= learning_rate * dL_db1

      return loss

  #function to return only a subset of the trianing data so we dont use all of it at once
    def get_batch(self, X, Y, batch_size = 32):
      num_data_points = X.shape[0]
      indices = np.random.choice(num_data_points, batch_size)
      return X[indices], Y[indices]

    def train(self, X, Y, epochs, learning_rate):
      num_batches = X.shape[0] // 32

      for epoch in range(epochs):
        #lr = learning_rate * (0.001 ** (epoch // 10))
        for batch in range(num_batches):
          X_batch, Y_batch = self.get_batch(X, Y)
          self.forward_pass(X_batch)
          loss = self.backward_pass(X_batch, Y_batch, learning_rate)
        print("Epoch: " + str(epoch) + " Loss: " + str(loss))

    def predict(self, X):
      return np.argmax(self.forward_pass(X))

    def accuracy(self, X, Y):
      Y_hat = self.predict(X)
      Y_hat = np.argmax(Y_hat, axis=1)
      return np.mean(Y_hat == Y)
    
    def save(self, filepath):
      np.savez(
        filepath,
        w1=self.w1,
        b1=self.b1,
        w2=self.w2,
        b2=self.b2,
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        output_size=self.output_size
      )

    @staticmethod
    def load(filepath):
        data = np.load(filepath)
        nn = NeuralNetwork(
            int(data["input_size"]),
            int(data["hidden_size"]),
            int(data["output_size"])
        )
        nn.w1 = data["w1"]
        nn.b1 = data["b1"]
        nn.w2 = data["w2"]
        nn.b2 = data["b2"]
        return nn


