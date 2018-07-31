
## Introduction <a id="toc"></a>

Today we're going to look at linear regression!

### Table of Contents
1. [Import Data](#data)
2. [Scatter Plot](#scatter)
3. [Feature Engineering: Calculating Slope](#slope)
4. [Histogram](#hist)
5. [Summary Statistics](#sumstats)
6. [Initial Model](#model1)
7. [Residual Sum of Squares](#rss)
8. [Error Functions](#error)
9. [Visualizing Loss](#vizloss)
9. [Gradient Descent](#grad_desc)

## 1. Import Data <a id="data"></a>
To start, we'll need to import some data in order to perform our regression.  
Import the 'movie_data.xlsx' file as a pandas DataFrame and assign it to the variable 'df'.


```python
#import the 'movie_data.xlsx' file as a pandas DataFrame and assign it to the variable df here.
```

## 2. Scatter Plot <a id="scatter"></a>  
Create a Scatter Plot of the budget and  Domestic Gross (domgross)


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
# Scatter Plot
```

Hopefully you see some (weak) correlation.

Let's start thinking about linear regression in a little more depth. 
Here, we have a simple 2 variable relation. You might remember the infamous equation y=mx+b from when you took Algebra. Here we're trying to determine how the Domestic Gross Sales is related to the movie's budget. 

x - budget
y - domestic gross sales

Let's also assume that b=0 in the equation y=m*x+b.

Thus we have y=m*x+0 and solving for m, y/x=m.

Let's investigate this relationship further.

## 3. Feature Engineering: Calculating Slope <a id="slope"></a>
Create a new column m=y/x, the ratio of a movie's domestic gross sales to it's budget.


```python
df['m'] = #write a formula to create the new column
```

## 4. Histogram <a id="hist"></a>  
Let's investigate the slope a bit more statistically.
Visualize the distribution of m using a histogram.


```python
 #Write code to display m as a histogram
```

## 5. Summary Statistics <a id="sumstats"></a> . 
Calculate the mean and median of m.


```python
mean = #your code here
median = #your code here
print('Mean: {}, Median: {}'.format(mean, median))
```

## 6. Initial Model <a id="model1"></a>
Create two initial models using these summary statistics.  
Each will be a linear model using that summary statistic to predict the gross domestic sales using a movie's budget.
Plot the data as a scatter plot and then draw each of these predictive models on top of the data. Be sure to include a title and use appropriate labels for the x and y axis.


```python
#Starter code
#In order to graph y = 1.575*x and y = 1.331*x,
#we'll generate a series of x-values and then calculate their associated y-values.
#Here's some x values to get you started.
x = np.linspace(start=df.budget.min(), stop=df.budget.max(), num=10**5)

#Calculate their corresponding y-values and plot the results on your graph.
#Don't forget to also graph the original data.

#Visual code here
#Include title, xlabel, ylabel, and legend ~7 lines total
```

## 7. Residual Sum of Squares <a id="rss"></a>
### a. Write an error function to calculate the residual sum of squares for a given model.  
Your function should take in 3 inputs:
 * a list of x values
 * a list of y values (corresponding to the x values passed)
 * a list of $\hat{y}$ values produced by the model (corresponding to the x values passed)


### b. Now use your residual sum of squares function to evaluate each of the previous 2 models.


```python
#Your code here
```

## Error/Loss Functions

From this you should see that the median ratio model produces a lower residual sum of squares. As such, this would be evaluated as our superior model of the two. In machine learning, we do just that; we provide an error or loss function to the learning algorithm which will then produce a model to minimize this error or loss.

In this linear regression problem, we are looking for which m will produce the minimum residual sum of squares.

Given,

$\hat{y} = m*x$

Minimize  
$ \sum(\hat{y}-y)^2$

### 8. Write a function to calculate the rss for a given slope m. <a id="error"></a>



```python
def error(x, y, m):
    return error
```

### Gradient Descent

Now finally to use this function to find our optimal model!
Here we'll use gradient descent. From calculus, you may recall that when working on optimization problems, we often turn to derivatives. Recall that the derivative of a function at a given point is the slope of the line tangent and the rate of change at that point. As a result, whenever we have a minimum or maximum the derivative will be zero:

![](./images/maxmin2.gif)


The idea of gradient descent is we take smaller and smaller steps downhill, converging in upon a minimum. There are some caveats to this process, such as finding local minimum rather then global, but the process helps guide our search in navigating an n-dimensional space for an optimal solution.

### 9. Visualizing the Loss Function <a id="vizloss"></a>
a. Create a range of reasonbale values for m.  
b. Then calculate their associated rss scores using your error function.  
c. Plot them on a graph.  


```python
#Scatter Plot Code Here
```

### 10. Gradient Descent. <a id="grad_desc"></a>
Now it's time to put all of this together and write a gradient descent function.  
This is ultimately the tool we will use to tune our model and find the optimal solution.  
The function should take in 5 parameters:
1. error_function; use the previous error function you defined above. we intend to minimize this
2. step_size_coefficient; this will modify how large the steps we take are
3. precision;  This will be a small parameter >= 0. 
               If an iteration does not change the result of the error function
               by this amount or more the algorithm will terminate.
4. max_iterations; Terminate the algorithm after this number of iterations.
5. start_x; The original x-value to initialize gradient descent.

**Hint:** Use the np.gradient() function to calculate the derivative at a given point for each iteration.


```python
def gradient_descent(error_function, step_size_coeff, max_iterations, start_x):
    #Step 1 create a while loop that executes until the difference between one iteration and the next is less then the precision value
    #Step 2 calculate the gradient (the derivative)
    #Step 3 take a step in that direction
    #Iterate!
```

## [Back to Top](#toc)
