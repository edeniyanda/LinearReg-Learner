# Import the required libraries
import numpy as np 
import matplotlib.pyplot as plt

# Define the training data points
x_train = np.array([1.0, 2.0])
y_train = 2*x_train + 4

# Set the initial values for the slope and the intercept
w = 3
b = 4.5


def y_hat(x ,y, w, b):
    yhat = np.zeros
# Define the cost function that computes the mean squared error
def compute_cost(x, y, w, b):
    m = x.shape[0]    # Get the number of training examples
    cost = 0          # Initialize the cost to zero
    
    # Compute the cost for each training example
    for i in range(m):
       f_wb = (w * x[i] + b) - y[i]   # Compute the predicted value for the current example
       cost += (f_wb) ** 2            # Add the squared error to the cost
    
    total_cost = (1 / (2 * m)) * cost  # Compute the average cost
    
    return total_cost


# Define the gradient function that computes the partial derivatives of the cost function with respect to w and b
def compute_gradient(x,y,w,b):
    m = x.shape[0]    # Get the number of training examples
    djdw = 0           # Initialize the partial derivative of the cost with respect to w to zero
    djdb = 0           # Initialize the partial derivative of the cost with respect to b to zero
    
    # Compute the partial derivatives of the cost function for each training example
    for i in range(m):
        f_wb = w * x[i] + b             # Compute the predicted value for the current example
        djdwi = (f_wb - y[i]) * x[i]    # Compute the partial derivative of the cost with respect to w
        djdbi = (f_wb - y[i])           # Compute the partial derivative of the cost with respect to b
        djdw += djdwi                   # Add the partial derivative to the total partial derivative with respect to w
        djdb += djdbi                   # Add the partial derivative to the total partial derivative with respect to b
    
    djdw = djdw * (1 / m)               # Compute the average partial derivative with respect to w
    djdb = djdb * (1 / m)               # Compute the average partial derivative with respect to b
    
    return djdw, djdb


# Set the learning rate
learning_rate = 0.09

sgd = 2
# Train the model
for iter in range(400):
    cost = compute_cost(x_train, y_train, w, b)    # Compute the cost for the current parameters
    
    # Print the current cost and the current parameter values
    print(f'{iter}: cost:{cost}, parameter: w = {w}, b = {b}')
    
    dw, db = compute_gradient(x_train, y_train, w, b)   # Compute the gradients with respect to w and b
    
    # Update the slope and intercept based on the gradients and the learning rate
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    if cost < sgd:
        bw = w
        bb = b
        

print(f"Best parameters: w = {bw}, b = {bb}")



# Visualize the trained model
plt.plot(x_train, y_hat(x_train, bw, bb), label="Training Model", color="red")
plt.scatter(x_train, y_train, c="g", marker="+", label="Training Data")
plt.title("A sample Training model")
plt.xlabel("Feature")
plt.ylabel("Label")
plt.legend()
plt.show()

# if __name__ == "__main__":
#     print(compute_cost(x_train, y_train, w, b))