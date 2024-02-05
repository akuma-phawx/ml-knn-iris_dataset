import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.style.use("ggplot")

    # x_train is the input variable (size in 1000 square feet)
    x_train = np.array([1.0, 2.0])
    # y_train is the target (price in 1000s of dollars)
    y_train = np.array([300.0, 500.0])

    w = 200
    b = 100
    x_i = 1.2
    cost_1200sqft = w * x_i + b
    print(f"Cost of 1200 sq ft: {cost_1200sqft}")
   
    tmp_f_wb = compute_model_output(x_train, w, b,)
    # Plot our model prediction
    plt.plot(x_train, tmp_f_wb, label="Model Prediction")
    # Plot the data points
    plt.scatter(x_train, y_train, marker="x", c="r")
    # Set the title
    plt.title("Housing Prices")
    # Set the y-axis label
    plt.ylabel("Price in $1000s")
    # Set the x-axis label
    plt.xlabel("Size in ft^2")

    plt.show()

    print(compute_cost(x_train, y_train, 100, 2))
    
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    
    return f_wb


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray (m,)): Data, m examples 
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters  

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
                to fit the data points in x and y
    """
    
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost     
    return cost_sum / (2*m)
    
    
if __name__ == "__main__":
    main()