
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
import pandas as pd
df = pd.read_excel("movie_data.xlsx")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>domgross</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13000000</td>
      <td>25682380</td>
      <td>21 &amp;amp; Over</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45658735</td>
      <td>13414714</td>
      <td>Dredd 3D</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000000</td>
      <td>53107035</td>
      <td>12 Years a Slave</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61000000</td>
      <td>75612460</td>
      <td>2 Guns</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40000000</td>
      <td>95020213</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Scatter Plot <a id="scatter"></a>  
Create a Scatter Plot of the budget and  Domestic Gross (domgross)


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
plt.scatter(df.budget, df.domgross)
```




    <matplotlib.collections.PathCollection at 0x11836f978>




![png](index_files/index_6_1.png)


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
df['m'] = df.domgross/df.budget #write a formula to create the new column
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>domgross</th>
      <th>title</th>
      <th>m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13000000</td>
      <td>25682380</td>
      <td>21 &amp;amp; Over</td>
      <td>1.975568</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45658735</td>
      <td>13414714</td>
      <td>Dredd 3D</td>
      <td>0.293804</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000000</td>
      <td>53107035</td>
      <td>12 Years a Slave</td>
      <td>2.655352</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61000000</td>
      <td>75612460</td>
      <td>2 Guns</td>
      <td>1.239549</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40000000</td>
      <td>95020213</td>
      <td>42</td>
      <td>2.375505</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Histogram <a id="hist"></a>  
Let's investigate the slope a bit more statistically.
Visualize the distribution of m using a histogram.


```python
df.m.hist() #Write code to display m as a histogram
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c87ad68>




![png](index_files/index_12_1.png)


## 5. Summary Statistics <a id="sumstats"></a> . 
Calculate the mean and median of m.


```python
mean = df.m.mean()
median = df.m.median()
print('Mean: {}, Median: {}'.format(mean, median))
```

    Mean: 1.574873371518024, Median: 1.3310858237179488


## 6. Initial Model <a id="model1"></a>
Create two initial models using these summary statistics.  
Each will be a linear model using that summary statistic to predict the gross domestic sales using a movie's budget.
Plot the data as a scatter plot and then draw each of these predictive models on top of the data. Be sure to include a title and use appropriate labels for the x and y axis.


```python
import numpy as np
```


```python
#Starter code
#In order to graph y = 1.575*x and y = 1.331*x,
#we'll generate a series of x-values and then calculate their associated y-values.
#Here's some x values to get you started.
x = np.linspace(start=df.budget.min(), stop=df.budget.max(), num=10**5)

#Calculate their corresponding y-values and plot the results on your graph.
#Don't forget to also graph the original data.

#Visual code here
plt.scatter(x, 1.575*x, label='Mean Ratio Model')
plt.scatter(x, 1.331*x, label='Median Ratio Model')
plt.scatter(df.budget, df.domgross, label='Actual Data Points')
plt.title('Gross Domestic Sales vs. Budget', fontsize=20)
plt.xlabel('Budget', fontsize=16)
plt.ylabel('Gross Domestic Sales', fontsize=16)
plt.legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x11ccaa588>




![png](index_files/index_17_1.png)


## 7. Residual Sum of Squares <a id="rss"></a>
### a. Write an error function to calculate the residual sum of squares for a given model.  
Your function should take in 3 inputs:
 * a list of x values
 * a list of y values (corresponding to the x values passed)
 * a list of $\hat{y}$ values produced by the model (corresponding to the x values passed)


### b. Now use your residual sum of squares function to evaluate each of the previous 2 models.


```python
df['1.575x'] = df.budget.astype(float)*1.575
df['1.331x'] = df.budget.astype(float)*1.331
# df.head()
def rss(residual_col):
    return sum(residual_col.astype(float).map(lambda x: x**2))
for col in ['1.575x','1.331x']:
    print('Residual Sum of Squares for {}: {}'.format(col, rss(df[col])))
df.head()
```

    Residual Sum of Squares for 1.575x: 5.473002195341657e+17
    Residual Sum of Squares for 1.331x: 3.908594504280841e+17





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>domgross</th>
      <th>title</th>
      <th>m</th>
      <th>1.575x</th>
      <th>1.331x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13000000</td>
      <td>25682380</td>
      <td>21 &amp;amp; Over</td>
      <td>1.975568</td>
      <td>2.047500e+07</td>
      <td>1.730300e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45658735</td>
      <td>13414714</td>
      <td>Dredd 3D</td>
      <td>0.293804</td>
      <td>7.191251e+07</td>
      <td>6.077178e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000000</td>
      <td>53107035</td>
      <td>12 Years a Slave</td>
      <td>2.655352</td>
      <td>3.150000e+07</td>
      <td>2.662000e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61000000</td>
      <td>75612460</td>
      <td>2 Guns</td>
      <td>1.239549</td>
      <td>9.607500e+07</td>
      <td>8.119100e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40000000</td>
      <td>95020213</td>
      <td>42</td>
      <td>2.375505</td>
      <td>6.300000e+07</td>
      <td>5.324000e+07</td>
    </tr>
  </tbody>
</table>
</div>



## Error/Loss Functions

From this you should see that the median ratio model produces a lower residual sum of squares. As such, this would be evaluated as our superior model of the two. In machine learning, we do just that; we provide an error or loss function to the learning algorithm which will then produce a model to minimize this error or loss.

In this linear regression problem, we are looking for which m will produce the minimum residual sum of squares.

Given,

$\hat{y} = m*x$

Minimize  
$ \sum(\hat{y}-y)^2$

### 8. Write a function to calculate the rss for a given slope m. <a id="error"></a>



```python
def rss(residual_col):
    return sum(residual_col.astype(float).map(lambda x: x**2))
```


```python
def error(x, y, m):
    model = m * x
    residuals = model - y
    total_rss = rss(residuals)
    return total_rss
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


```python
ms = []
errors = []
for m in np.linspace(start=0.5, stop=1.5, num=101):
    ms.append(m)
    errors.append(error(df.budget, df.domgross, m))
plt.scatter(ms ,errors)
plt.scatter(ms[58], errors[58], color='red')
print(ms[58])
```

    1.08



![png](index_files/index_27_1.png)


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
cur_x = 1.5 # The algorithm starts at x=2
gamma = 1*10**(-7) # step size multiplier
print(gamma)
precision = 0.0000000001
previous_step_size = 1 
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter

while (previous_step_size > precision) & (iters < max_iters):
    print('Current value: {} RSS Produced: {}'.format(cur_x, error(df.budget, df.domgross, cur_x)))
    prev_x = cur_x
    x_survey_region = np.linspace(start = cur_x - previous_step_size , stop = cur_x + previous_step_size , num = 101)
    rss_survey_region = [np.sqrt(error(df.budget, df.domgross, m)) for m in x_survey_region]
    gradient = np.gradient(rss_survey_region)[50] 
    cur_x -= gamma * gradient #Move opposite the gradient
    previous_step_size = abs(cur_x - prev_x)
    iters+=1

print("The local minimum occurs at", cur_x)
#The output for the above will be: ('The local minimum occurs at', 2.2499646074278457)
```

    1e-07
    Current value: 1.5 RSS Produced: 2.6084668957174013e+17
    Current value: 1.1330655714424849 RSS Produced: 2.217773053377032e+17
    Current value: 1.1131830522749038 RSS Produced: 2.2135715390729427e+17
    Current value: 1.1124754156940968 RSS Produced: 2.2134541499866915e+17
    Current value: 1.1124506992634804 RSS Produced: 2.213450089740645e+17
    Current value: 1.1124498365366668 RSS Produced: 2.2134499480664778e+17
    Current value: 1.1124498064238966 RSS Produced: 2.2134499431215165e+17
    Current value: 1.1124498053728342 RSS Produced: 2.2134499429489165e+17
    The local minimum occurs at 1.1124498053361447


## [Back to Top](#toc)
