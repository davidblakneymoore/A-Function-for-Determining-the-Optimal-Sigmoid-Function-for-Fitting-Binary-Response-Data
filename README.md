# A-Function-for-Determining-the-Optimal-Sigmoid-Function-for-Fitting-Binary-Response-Data

Though logistic regression is almost exclusively used for modeling probabilities based on binary response data, there are many other options. This function fits ten separate models to binary response data and determines which of the ten are best based on the residual sums of squares. The ten model types are based on the logistic function, the hyperbolic tangent, the arctangent function, the Gudermannian function, the error function, a generalised logistic function, an algebraic function, a more general algebraic function, the Gompertz function, and the Gompertz function after it has been rotated by 180 degrees. All ten of these functions have been rescaled so that they are bounded by 0 and 1 on the response variable axis. Please note that the Gompertz function and the rotated Gompertz function are the only functions out of the ten that are not radially symmetric about their inflection points - one side of these functions approaches the asymptote more gradually than the other.

This function uses the `R2jags` package heavily and it returns all the pertinent information for each model. For each model, it returns the model name, the actual model (with the parameters that best fit the data included), a data frame of fitted predictors and responses for plotting, the residual sum of squares, and the output from the Bayesian analysis, which includes all the coefficients and the Bayesian p value.

Unfortunately, this function can't handle multiple predictor variables yet. Someday I may update it to be able to account for multiple predictor variables.

I took an outstanding course in Bayesian statistics with Dr. Remington Moll at the University of New Hampshire that helped me write this function.

This function takes 8 arguments. The first two are required.

`Predictor` is a vector of predictor variables to be used in the model.

`Response` is a vector of response variables to be used in the model. It should be binary (`1`s and `0`s).

`Data_Frame` is an optional data frame to include such that column names can be supplied for the `Predictor` and the `Response` arguments. The data frame that these columns are from should be provided for this Data_Frame argument.

`Number_of_Iterations = 100000` is the number of iterations each Markov chain Monte Carlo simulation undergoes. The default value for this argument is `100000`. More iterations lead to more precise results, but the program will take longer to run. I've found that doing 100000 iterations generally allows all the models to converge - doing 10000 generally isn't quite enough.

`Thinning_Rate = 1` describes how many iterations of the Markov chain Monte Carlo simulation are performed per stored value. The default value for this argument is `1`. For example, if the thinning rate is 3, every third iteration of a Markov chain Monte Carlo simulation would be stored as model output.

`Burn_in_Value = 1000` is the number of initial iterations of each Markov chain Monte Carlo simulation that are discarded. The default value of this argument is `1000`. Typically, it takes several iterations for parameter estimates to stabilize, so it is worthwhile to discard the first several iterations so they are not used in the final parameter estimates. It is often worth looking at plots of how parameter estimates change with Markov chain Monte Carlo iteration to ensure enough initial iterations are discarded and parameter estimates stabilize properly.

`Number_of_Chains = 3` is the number of separate Markov chain Monte Carlo simulations that will be run. The default, `3`, is common, and fewer than 3 is not recommended.

`Working_Directory = getwd()` is the working directory in which to save the .txt files used by the `R2jags::jags()` function.

<b>Works Cited</b>

Su, Y.-S., and M. Yajima. 2021. R2jags: Using R to Run 'JAGS'. R package version 0.7-1. https://cran.r-project.org/web/packages/R2jags/.
