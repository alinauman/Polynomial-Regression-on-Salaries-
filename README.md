# Polynomial Regression

## Algorithm
In statistics, polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as an nth degree polynomial in x. 
Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y |x). Although polynomial regression fits a nonlinear model to the data, as a statistical estimation problem it is linear, in the sense that the regression function E(y | x) is linear in the unknown parameters that are estimated from the data. 
For this reason, polynomial regression is considered to be a special case of multiple linear regression.

Polynomial regression can find the relationship between input features and the output variable in a better way even if the relationship is not linear. 

![Image of Original vs Predicted Salary](https://github.com/alinauman/Polynomial-Regression-on-Salaries-/blob/master/Original_vs_Predicted_Salary.png)

Our prediction does not exactly follow the trend of salary but it is close. Linear regression can only return a straight line. But in polynomial regression, we can get a curved line like that. 
If the line would not be a nice curve, polynomial regression can learn some more complex trends as well.

![Image of Cost per Epoch](https://github.com/alinauman/Polynomial-Regression-on-Salaries-/blob/master/Cost_vs_Epoch.png)

The cost fell drastically in the beginning and then the fall was slow. In a good machine learning algorithm, cost should keep going down until the convergence.

Tutorial URL:[Link][https://towardsdatascience.com/polynomial-regression-from-scratch-in-python-1f34a3a5f373]
