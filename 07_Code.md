# Derivatives

- Randomly changing and searching for optimal weights and biases did not prove fruitful for one main reason: the number of possible combinations of weights and biases is infinite, and we need something smarter than pure luck to achieve any success. 
- Each weight and bias may also have different degrees of influence on the loss — this influence depends on the parameters themselves as well as on the current sample, which is an input to the first layer. 
- These input values are then multiplied by the weights, so the input data affects the neuron’s output and affects the impact that the weights make on the loss. 
- The same principle applies to the biases and parameters in the next layers, taking the previous layer’s outputs as inputs. This means that the impact on the output values depends on the parameters as well as the samples — which is why we are calculating the loss value per each sample separately. 
- Finally, the function of ​how​ a weight or bias impacts the overall loss is not necessarily linear. In order to know ​how​ to adjust weights and biases, we first need to understand their impact on the loss.
- One concept to note is that we refer to weights and biases and their impact on the loss function. The loss function doesn’t contain weights or biases, though. The input to this function is the output of the model, and the weights and biases of the neurons influence this output. Thus, even though we calculate loss from the model’s output, not weights/biases, these weights and biases directly impact the loss

### The Impact of a Parameter on the Output

Taken an example of f =2(x), what is the impact on Output.

#### The Slope

- f(x) = 2x function, which is a line. 

**How might you define the impact that x will have on y?** 

- Some will say, “y is double x” 
- Another way to describe the impact of a linear function such as this comes from algebra: the slope. 
- The slope of a line is: 
      slope of a line = change in y / change in x = Δ y / Δ x
    - It is change in y divided by change in x, or in math — delta y divided by delta x

**What’s the slope of f(x) = 2x then?**

- To calculate the slope, first we have to take any two points lying on the function’s graph and subtract them to calculate the change. 
- Subtracting the points means to subtract their x and y dimensions respectively. 
- Division of the change in y by the change in x returns the slope


```python
import matplotlib.pyplot as plt
import numpy as np

def f(x):
  return 2 * x

x = np.array(range(5))
y = f(x)

print(x)
print(y)

plt.plot(x,y)
plt.show()
```

    [0 1 2 3 4]
    [0 2 4 6 8]



    
![png](output_2_1.png)
    



```python
slope = ((y[1]-y[0]) / (x[1]-x[0]))
print("Slope of the line 0 and 1:",slope)

slope = ((y[2]-y[1]) / (x[2]-x[1]))
print("Slope of the line 1 and 2:",slope)

slope = ((y[2]-y[0]) / (x[2]-x[0]))
print("Slope of the line 0 and 2:",slope)

# Inference :
# Slope is same for Linear Functions
```

    Slope of the line 0 and 1: 2.0
    Slope of the line 1 and 2: 2.0
    Slope of the line 0 and 2: 2.0


- It is not surprising that the slope of this line is 2. We could say the measure of the impact that ​x has on ​y​ is 2. 
- We can calculate the slope in the same way for any linear function, including linear functions that aren’t as obvious.


```python
# Non Linear Functions
import matplotlib.pyplot as plt
import numpy as np

def f(x):
  return 2 * x ** 2 # 2x^2, 2 x power 2 which is not linear

x = np.array(range(5))
y = f(x)

print(x)
print(y)

plt.plot(x,y)
plt.show()
```

    [0 1 2 3 4]
    [ 0  2  8 18 32]



    
![png](output_5_1.png)
    



```python
slope = ((y[1]-y[0]) / (x[1]-x[0]))
print("Slope of the line 0 and 1:",slope)

slope = ((y[2]-y[1]) / (x[2]-x[1]))
print("Slope of the line 1 and 2:",slope)

slope = ((y[2]-y[0]) / (x[2]-x[0]))
print("Slope of the line 0 and 2:",slope)

# Inference :
# Slope is not same for Non Linear Functions
```

    Slope of the line 0 and 1: 2.0
    Slope of the line 1 and 2: 6.0
    Slope of the line 0 and 2: 4.0


**How might we measure the impact that x has on y in this nonlinear function?**

- Calculus proposes that we measure the slope of the tangent line at x (for a specific input value to the function), giving us the instantaneous slope (slope at this point), which is the derivative.
- The tangent line is created by drawing a line between two points that are “infinitely close” on a curve, but this curve has to be differentiable at the derivation point.
- This means that it has to be continuous and smooth (we cannot calculate the slope at something that we could describe as a “sharp corner,” since it contains an infinite number of slopes).
- Then, because this is a curve, there is no single slope.
- Slope depends on where we measure it.

To give an immediate example, we can approximate a derivative of the function at x by using this point and another one also taken at x, but with a very small delta added to it, such as 0.0001.

- This number is a common choice as it does not introduce too large an error (when estimating the derivative) or cause the whole expression to be numerically unstable (x might round to 0 due to floating-point number resolution). 
- This lets us perform the same calculation for the slope as before, but on two points that are very close to each other, resulting in a good approximation of a slope at x:


```python

```
