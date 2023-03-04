# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the input feature and output label for the training data
x_train = np.array([2,5,6,7,8,11,14,16,18,20,24])
y_train = np.array([12,17,20,25,29,32,36,43,51,54,60])

# Set the initial values of slope and intercept
w = 3
b = 5

# Define a function to compute the predicted output based on the input feature, slope and intercept
def y_hat(x, w, b):
    return w * x + b

# Define a function to compute the mean squared error between the predicted output and the actual output
def cost_function(x, y, w, b):
    y_pred = y_hat(x_train, w, b)
    return np.mean(np.square( y_pred - y))

# Set the learning rate for the gradient descent algorithm
learning_rate = 0.01   

# Run the gradient descent algorithm for 400 iterations
for i in range(400):
    # Compute the gradients of the cost function with respect to the slope and intercept
    dw = np.mean((y_hat(x_train, w, b) - y_train) * x_train)
    db = np.mean((y_hat(x_train, w, b) - y_train))
    
    # Update the slope and intercept based on the gradients and the learning rate
    w = w - dw * learning_rate
    b = b - db * learning_rate
    
    # Compute the cost function and print it to monitor the progress of the optimization
    cost = cost_function(x_train, y_train, w, b)
    print(f"Iteration {i}: Cost = {cost}, w = {w}, b = {b}")
       
# Visualize the trained model
plt.plot(x_train, y_hat(x_train, w, b), label="Training Model", color="red")
plt.scatter(x_train, y_train, c="g", marker="+", label="Training Data")
plt.title("A sample Train model")
plt.xlabel("Feature")
plt.ylabel("Label")
plt.legend()
plt.show()
