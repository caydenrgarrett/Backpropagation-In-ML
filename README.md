# Backpropagation in 11 Lines

This project explores the implementation of simple neural networks using **Python** and the **NumPy** library.  
It demonstrates the basic principles of building and training neural networks with one and two hidden layers.

---

## 🔹 Bilayered Neural Network

This section implements a basic neural network with a single hidden layer. <br>

![image alt](https://ljvmiranda921.github.io/assets/png/cs231n-ann/archi.png)

### Code Explanation
- **`nonlin(x, deriv=False)`**: Sigmoid activation function. Returns the sigmoid of the input `x` or its derivative if `deriv=True`.  
- **`X`**: The input dataset (rows = training examples, columns = features).  
- **`y`**: The output dataset (rows = expected outputs).  
- **`syn0`**: Synaptic weights connecting the input layer to the hidden layer (randomly initialized).  
- **Training Loop**: Runs for 10,000 iterations to train the network.  
- **Forward Propagation**: Input `l0` × weights `syn0` → passed through sigmoid → hidden layer output `l1`.  
- **Error Calculation**: Difference between expected output `y` and predicted output `l1`.  
- **Delta Calculation**: `l1_error` × derivative of sigmoid(`l1`) → `l1_delta`, which indicates how much to adjust weights.  
- **Weight Update**: `syn0` updated with the dot product of `l0.T` and `l1_delta`.

Code:
```python
import numpy as np

# Sgmoid function
def nonlin(x, deriv=False):
  if (deriv==True):
    return x * (1-x)
  return 1 / (1+np.exp(-x))

# Input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# Output dataset
y = np.array([[0,0,1,1]]).T

np.random.seed(1)

syn0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):

  # Forward Propagation
  l0 = X
  l1 = nonlin(np.dot(l0, syn0))

  # By how much was missed?
  l1_error = y - l1

  # Multiply how much we missed,
  # by the slope of the sigmoid values
  l1_delta = l1_error * nonlin(l1, True)
  syn0 += np.dot(l0.T,l1_delta)

print("Output after training:")
print(l1)
```
Output:
```
Output after training:
[[0.00966449]
 [0.00786506]
 [0.99358898]
 [0.99211957]]
```

---

## 🔹 Trilayered Neural Network

This section implements a neural network with **two hidden layers**. <br>

![image alt](https://cdn-media-1.freecodecamp.org/images/FDWrPCgJTJbH3MPUSyT0tgG2Zi2TYczZDOAj)

### Code Explanation
- **`nonlin(x, deriv=False)`**: Same sigmoid activation function as before.  
- **`X`**: Input dataset.  
- **`y`**: Output dataset.  
- **`syn0`**: Synaptic weights connecting the input layer to the first hidden layer.  
- **`syn1`**: Synaptic weights connecting the first hidden layer to the second hidden layer (both initialized randomly).  
- **Training Loop**: Runs for 60,000 iterations to train the network.  
- **Forward Propagation**:  
  - `l0` → first hidden layer via `syn0` → output `l1`  
  - `l1` → second hidden layer via `syn1` → final output `l2`  
- **Error Calculation**: `l2_error` = `y - l2` (difference between expected and predicted outputs).  
- **Error Reporting**: Mean absolute error printed every 10,000 iterations.  
- **Backpropagation**: Error propagated backward → `l2_delta` and `l1_delta` computed.  
- **Weight Update**: `syn1` and `syn0` updated using the deltas and outputs from previous layers.

Code:
```python
import numpy as np

def nonlin(x, deriv=False):
  if(deriv==True):
    return x * (1 - x)
  return 1/ (1 + np.exp(-x))

# Input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# Output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

# Randomly inititalize our weights with a mean of zero
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):
  # Feed forward through layers 0 through 2
  l0 = X
  l1 = nonlin(np.dot(l0,syn0))
  l2 = nonlin(np.dot(l1,syn1))

  # By how much did we miss our target value
  l2_error = y - l2

  if (j%10000) == 0:
    print("Error:", str(np.mean(np.abs(l2_error)))) # Changed + to , and made it a single argument


  # Which direction is the actual target value,
  # if unsure don't change that much
  l2_delta = l2_error*nonlin(l2,deriv=True)

  # How much did each l1 value contribute to the l2 error (according to weights)
  l1_error = l2_delta.dot(syn1.T)
  l1_delta = l1_error * nonlin(l1,deriv=True)

  # update weights
  syn1 += l1.T.dot(l2_delta)
  syn0 += l0.T.dot(l1_delta) # Changed l0.dot to l0.T.dot

print("\n2nd output after training:") # Added print statement for final output
print(l2)
```
Output:
```
Error: 0.4964100319027255
Error: 0.008584525653247157
Error: 0.0057894598625078085
Error: 0.004629176776769985
Error: 0.0039587652802736475
Error: 0.003510122567861678

2nd output after training:
[[0.00260572]
 [0.99672209]
 [0.99701711]
 [0.00386759]]
```

---

## 📌 Summary

This notebook provides a **basic introduction** to building and training simple neural networks from scratch, including both a single-hidden-layer (bilayered) and a two-hidden-layer (trilayered) implementation.
