Lecture 9
=========


Classification and Regression
-----------------------------

In supervised learning, we are primarily interested in *classification* (also know as *logistic regression*) and *regression* problems.

Classification deals with *categories*; it's output values are non-parametric.
An example could be a model that classifies handwriting samples, as belonging to the `1`, `2`, `3`, ... categories.

Regression deals with *real values*, that is parametric output.
An example could be predicting the temperature, house prices, etc.


Regression and Gradient Descent
-------------------------------

.. warning:: There are a number of assignments associated with this lecture.
    When doing the assignments, it is important that you take a programmatic approach rather than just fiddling with the code.
    The concepts in this lecture are essential to working with machine learning models, so try work with it as much as possible.

What is Regression
------------------

In Data Science we make use of the term *regression* similarly to the use of the word in statistics:

.. glossary::

   Regression
      A measure of the relation between the mean value of one variable (e.g. output) and corresponding values of other variables (e.g. time vs cost).

What we usually mean when we are talking about regression is that we are trying to make a model for one or more parametric variables based on a set of other variables.

Simple Regression
-----------------

Let's look at a simple example:

.. figure:: _static/simple-regression.png
    :align: center
    :alt: Simple Regression
    :figclass: align-center

Above is a very small data set with four observations.
We don't really care what the data means for now, this only serves as an example.

The data set looks like this in Python:

.. code-block:: python

    x = [1, 3, 4.5, 5.5]
    y = [2.5, 3, 3, 3.5]

This represents the observations :math:`(1, 2.5), (3, 3), (4.5, 3), (5.5, 3.5)`.

Say we want to predict values from this data set.
That is, if I ask *for x = 2, what is y?*.
Looking at the plot above, we assume that we are looking at a linear relationship.
A linear relationship takes the form:

.. math::

    y = ax + b

Just as you learned in secondary school.

Our first attempt will be *the Normal Equation*:

.. math::

    \theta = (X^TX)^{-1}X^TY

The Normal Equation works with matrices, but we can easily work with matrices in Numpy:

.. code-block:: python
    :linenos:

    m = len(y)

    X = np.zeros((m, 2))

    X[: ,0] = 1
    X[: ,1] = x

We are inserting a left-most column of ones, because it would otherwise be a column vector rather than a matrix.
The one values are neutral to the calculations we will do on the matrix.
We get an `X` matrix looking like this:

.. code-block:: python

    array([[1. , 1. ],
           [1. , 3. ],
           [1. , 4.5],
           [1. , 5.5]])

First column is the ones, second column is the `x` (lowercase) from our data set.

We must also represent `y` as a Numpy array, and then we can use the Normal Equation:

.. code-block:: python

    y = np.array(y)
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

We now have a value for theta:

.. code-block:: python

    array([2.31521739, 0.19565217])

The theta value we calculated gives us a hypothesis function `h`:

.. math::

    h_\theta(x) = \theta_0 + \theta_1x \approx 2.315 + 0.196x

As you can see (right-most), this is a simple linear equation.
Let's try plot the line with our data set:

.. code-block:: python

    pyplot.scatter(x, y)
    a = np.linspace(0, 6, 2)
    b = theta[0] + a * theta[1]
    pyplot.plot(a, b)

.. figure:: _static/simple-regression-fit.png
    :align: center
    :alt: Simple Regression
    :figclass: align-center

As you can see, the line does not fit the individual observations, but it does fit the data set very well.
In fact, the Normal Equation gives us an *exact solution*.

However, there is a problem with the Normal Equation.
It worked well for our small data set, but it does not scale well with very large data sets.
The reason is that the Normal Equation make use of an inverse matrix operation (the -1), which is very expensive computationally.
Another problem with the Normal Equation is that it only works for straight lines (though it can be extended to polynomial functions in general) - we need something more general.

Therefore, we will look at other attempts.

Linear Approximation using Gradient Descent
-------------------------------------------

Our next attempt will scale much better with large data sets, however, it produces an *approximation* rather than an *exact solution*.
Very often that approximation is so good that there is practically no difference.

.. note:: The statement about *exact solutions* vs *aprroximated solutions* is typical for the difference between the field of *mathematics* and *engineering*. Because we are concerned with practical applications we are fine with an approximation (the eningneering approach).

.. note:: You are not expected to understand each and every detail of gradient descent.
    Focus on understanding the concepts, and the use of gradient descent.

Remember that we are trying to fit a hypothesis function `h` so that:

.. math::

    h = ax + b \approx y

That is: the hypothesis function `h` is a line that approximates `y`, where `y` is the ground truth.

However, we usually write the hypothesis function like this:

.. math::

    h_\theta = \theta_0 + \theta_1 x

So we have a function:

.. code-block:: python

    def h(theta, x):
        return theta[0] + theta[1] * x

We now want to optimize theta so it makes a good approximation of the line we fitted earlier.
We are going to use the *gradient descent* algorithm for this.
The algorithm consists of two functions - the `gradient_descent` function that loops over the data set, and the `gradient_step` function that handles one single data point.

.. code-block::

    def gradient_step(theta, x, y, alpha, verbose=False):
        if verbose: print("Gradient step ", theta, x, y, alpha)
        delta = np.zeros(np.shape(theta))
        m = len(y)
        for i in range(m):
            delta[0] -= (2/float(m)) * (y[i] - h(theta, x[i]))
            delta[1] -= (2/float(m)) * (y[i] - h(theta, x[i])) * x[i]
            if verbose: print(i, delta)
        if verbose:
            print("Theta", theta - alpha * delta)
            print("Cost", sum(1/(2*m) * np.square(h(theta, np.array(x)) - np.array(y))))
        return theta - alpha * delta

    def gradient_descent(x, y, initial_theta, alpha, iterations, verbose=False):
        theta = initial_theta
        for i in range(iterations):
            if verbose: print("** Iteration ", i)
            theta = gradient_step(theta, x, y, alpha, verbose)
        return theta

The `gradient_descent` function takes a number of parameters: `x` and `y` is the data set, `initial_theta` is the value that we use for theta to begin with, `alpha` is the *learning rate*, `iterations` is the number of times we are using the data set to optimize theta, and finally `verbose` lets us get a lot of output during processing.
Parameters such as `alpha` and `iterations` are know as *hyper-parameters* for the model.

.. glossary::

    Hyper-Parameter
        Parameters that affect the processing of a model.
        Think of the hyper-parameters as a configuration of the model.
        The hyper-parameters are not part of the result.

Let's give it a go:

.. code-block:: python

    gradient_descent(x, y, np.array([1, 2]), 0.01, 2000)

We get the output:

.. code-block:: python

    array([2.31401837, 0.19593298])

This is nearly a perfect match of the result we got from the Normal Equation.

The values that we used for initial theta, alpha, and iterations were pretty much taken out of thin air.
Still, we got to a reasonable result.

.. note:: **Assignment 7**

    Try change the hyper parameters for our first gradient descent implementation, and try to answer these questions:

    - How many iterations are necessary to get a good result?
    - What is the smallest and largest values of alpha that works well?
    - What influence does different values of theta have?

Optimizing Gradient Descent
---------------------------

The gradient descent implementation we saw above works fine, but it does not benefit from the fact that your processor probably have multiple cores, and supports SIMD.
Let's try to optimize the implementation.

We start with the hypothesis function:

.. code-block::

    def cost_2(theta, x, y):
        m = np.size(y)
        return sum(1/(2*m) * np.square(h2(theta, np.array(x)) - np.array(y)))

    def h2(theta, x):
        X = np.ones([len(x),len(theta)])
        X[:,1] = x
        return X.dot(theta.T)

    def linear_cost_prime(hyp, theta, x, y):
        m = np.size(y)
        delta = np.zeros(np.shape(theta))
        delta[0] -= (2/float(m)) * sum((y - hyp(theta, x)))
        delta[1] -= (2/float(m)) * sum((y - hyp(theta, x)) * x)
        return delta

Rather than working with one single value in the data set, the hypothesis function now works simultaneously on *all* values, because we perform the operations directly on the whole matrix.
For clarity, we also separated the cost function from the rest of the implementation.

The cost function calculates *how much of an error* our model makes.

We no longer have a `gradient_step` function, because our implementation is much simpler when we use SIMD.

.. code-block:: python

    def gradient_descent_2(hyp, cost, cost_prime, x, y, theta, alpha, iterations, verbose=False):
        cost_history = []
        delta = np.zeros(np.shape(theta))
        for i in range(iterations):
            if verbose: print("** Iteration ", i)
            delta = cost_prime(hyp, theta, x, y)
            theta = theta - alpha * delta
            cost_history.append(cost_2(theta, x, y))
        return theta, cost_history

Let's give it a go:

.. code-block:: Python

    theta, cost_history = gradient_descent_2(h2, cost_2, linear_cost_prime, x, y, np.array([0, 0]), 0.05, 5000)

We get pretty much the same result:

.. code-block:: Python

    array([2.31521739, 0.19565217])

You may have noticed that we also returned a cost history in our second implementation.

The cost function looks like this:

.. math::

    J_\theta(X) = -\frac{1}{m}\sum(h_\theta(X)-Y)^2

This is a cost function that we (most) often use when we optimize regression models.
It essentially sums up the squared error of the observations.

We can use that cost history for plotting the cost of approximation as a function of the number of iterations:

.. code-block:: python

    x_axis = np.linspace(0, 5000, 500)
    pyplot.plot(x_axis, cost_history[5:505])
    pyplot.title("Cost of approximation")
    pyplot.xlabel("Iterations")
    pyplot.ylabel("Cost")

.. figure:: _static/gd-cost-of-approx.png
    :align: center
    :alt: Simple Regression
    :figclass: align-center

As you can see, the cost doesn't decrease much after 1200 iterations - we are doing 5000 iterations, so we are wasting time here.
The problem is, we can't really tell how many iterations we need up front.

.. note:: **Assignment 8**

    How many iterations are necessary to get a good approximation?

    How much bigger is the error at that point than doing 5000 iterations?

.. note:: **Assignment 9**

    Our second implementation supports a `verbose` mode that allows you to observe gradual changes to the model during optimization.

    The problem is that the verbose mode is too verbose - it prints out thousands of lines of text optimizing a simple model.

    Add a parameter `verbose_interval` that will make the verbose mode less verbose.
    A value of 100 should print for every 100 optimizations etc.

    Try to do an optimization where you get 10 prints, and plot the suboptimal intermediate results in a figure.

.. note:: **Assignment 10**

    Find a data set of two variables that are linearly dependent.
    The data set should have tens of thousands of samples or more.

    Compare the two gradient descent implementations.
    How do the two implementations compare in computation time performance?

A Standard Implementation of Linear Regression
----------------------------------------------

We usually don't make low-level implementations of models the way we have done today.
However, doing it helps us understand a lot about how things work, even if we don't understand every single detail.
We will end of with a standard implementation of linear regression from the SciKit Learn machine learning library.

We are going to fit the same data set that we looked at earlier:

.. code-block:: Python

    x = [1, 3, 4.5, 5.5]
    y = [2.5, 3, 3, 3.5]
    X, y = np.array(x).reshape(-1, 1), np.array(y)

    from sklearn.linear_model import LinearRegression

    lg = LinearRegression().fit(X, y)
    lg.intercept_, lg.coef_

This will give us the output:

    (2.3152173913043477, array([0.19565217]))

As you can see, we get the same result as in our own implementation.

.. note:: **Assignment 11**

    Use SciKit Learn to make a model for the data set you worked on in Assignment 10.


Gradient Descent with Decision Boundary
---------------------------------------

Let's try another implementation of gradient descent that might be easier to understand, but would be difficult to use with SIMD.

This implementation clearly separates the linear model itself, and the gradient descent optimization algorithm.

You can also see a few plots of the decision boundary for the linear model.

.. raw:: html

   <iframe src="_static/gradient_descent.html" width="700px" height="500px"></iframe>


Using Tensorflow to Calculate Partial Derivatives
-------------------------------------------------

If you are very mathematically inclined you can take a look at how you can use Tensorflow to calculate partial derivatives.

This section is not part of the curriculum.


.. raw:: html

   <iframe src="_static/tf-partial-grad.html" width="700px" height="500px"></iframe>
