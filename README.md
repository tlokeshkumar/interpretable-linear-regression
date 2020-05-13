# Interpretable Linear Regression

A regular problem in multiple regression is asserting the relative influence of the predictors in the model. `Net Effects` is a well known technique that are used to measure the shares that each predictor have on the target variable in the coefficient of multiple determination R^2. In the case of correlated inputs, net effects fail to give interpretable results which calls for other positive interpretable metrics.

**Incremental Net Effect** is estimated as the marginal influence of the predictor in all possible coalation of predictors on the target variable. This idea directly relates to Shapley values in cooperative game theory.