
# A Function for Determining the Optimal Sigmoid Function for Fitting Binary
# Response Data

# David Moore
# davidblakneymoore@gmail.com
# May 2022


# The Explanation

# Though logistic regression is almost exclusively used for modeling
# probabilities based on binary response data, there are many other options.
# This function fits ten separate models to binary response data and
# determines which of the ten are best based on the residual sums of squares.
# The ten model types are based on the logistic function, the hyperbolic
# tangent, the arctangent function, the Gudermannian function, the error
# function, a generalised logistic function, an algebraic function, a more
# general algebraic function, the Gompertz function, and the Gompertz function
# after it has been rotated by 180 degrees. All ten of these functions have
# been rescaled so that they are bounded by 0 and 1 on the response variable
# axis. Please note that the Gompertz function and the rotated Gompertz
# function are the only functions out of the ten that are not radially
# symmetric about their inflection points - one side of these functions
# approaches the asymptote more gradually than the other.

# This function uses the 'R2jags' package heavily and it returns all the
# pertinent information for each model. For each model, it returns the model
# name, the actual model (with the parameters that best fit the data included),
# a data frame of fitted predictors and responses for plotting, the residual
# sum of squares, and the output from the Bayesian analysis, which includes all
# the coefficients and the Bayesian p value.

# Unfortunately, this function can't handle multiple predictor variables yet.
# Some day I may update it to be able to account for multiple predictor
# variables.

# I took an outstanding course in Bayesian statistics with Dr. Remington Moll
# at the University of New Hampshire that helped me write this function.

# This function takes 8 arguments. The first two are required.

# 'Predictor' is a vector of predictor variables to be used in the model.

# 'Response' is a vector of response variables to be used in the model. It
# should be binary (1s and 0s).

# 'Data_Frame' is an optional data frame to include such that column names can
# be supplied for the 'Predictor' and the 'Response' arguments. The data frame
# that these columns are from should be provided for this Data_Frame argument.

# 'Number_of_Iterations = 100000' is the number of iterations each Markov chain
# Monte Carlo simulation undergoes. The default is '1000'. More iterations lead
# to more precise results, but the program will take longer to run. I've found
# that doing 100000 iterations generally allows all models to converge - doing
# 10000 generally isn't enough.

# 'Thinning_Rate = 1' describes how many iterations of the Markov chain Monte
# Carlo simulation are performed per stored value. The default is '1'. For
# example, if the thinning rate is 3, every third iteration of the Markov chain
# Monte Carlo simulation would be stored as model output.

# 'Burn_in_Value = 1000' is the number of initial iterations of each Markov
# chain Monte Carlo simulation that are discarded. Typically, it takes several
# iterations for parameter estimates to stabilize, so it is worthwhile to
# discard the first several iterations so they are not used in the final
# parameter estimates. It is often worth looking at plots of how parameter
# estimates change with Markov chain Monte Carlo iteration to ensure enough
# initial iterations are discarded and parameter estimates stabilize properly.

# 'Number_of_Chains = 3' is the number of separate Markov chain Monte Carlo
# iterations that will be run. The default, '3', is common, and fewer than 3 is
# not recommended.

# 'Working_Directory = getwd()' is the working directory in which to save the
# .txt files used by the 'R2jags::jags()' function.


# The Function

Function_for_Fitting_an_Optimal_Sigmoid_Model <- function (Predictor, Response, Data_Frame, Number_of_Iterations = 100000, Thinning_Rate = 1, Burn_in_Value = 1000, Number_of_Chains = 3, Working_Directory = getwd()) {
  
  # Prepare the Data
  
  Predictor_Name <- deparse(substitute(Predictor))
  Response_Name <- deparse(substitute(Response))
  if (!missing(Data_Frame)) {
    if (class(Data_Frame) != 'data.frame') {
      stop ("'Data_Frame' must be of class 'data.frame'.")
    }
    Data_Frame <- Data_Frame[, c(Predictor_Name, Response_Name)]
  } else if (missing(Data_Frame)) {
    Data_Frame <- data.frame(Predictor = Predictor, Response = Response)
  }
  Predictor <- Data_Frame$Predictor
  Response <- Data_Frame$Response
  Data <- list(Predictor = Data_Frame$Predictor, Response = Data_Frame$Response, Number_of_Observations = nrow(Data_Frame))
  Initial_Values <- function () {
    list()
  }
  Fitted_Predictor_Values <- seq(Minimum_Predictor_Value, Maximum_Predictor_Value, length.out = 1000000)
  
  
  # Meet Some Initial Conditions
  
  if (!is.integer(Predictor) & !is.numeric(Predictor)) {
    stop ("The 'Predictor' argument must be of class 'integer' or 'numeric'.")
  }
  if ((!is.integer(Response) & !is.logical(Response) & !is.numeric(Response)) & ((all(Response %in% c(0, 1))) | (all(Response %in% c(T, F))))) {
    stop ("The 'Response' argument must be of class 'integer', 'logical', or 'numeric' and it must only contain '1's and '0's or 'TRUE's and 'FALSE's.")
  }
  if ((length(Number_of_Iterations) != 1) | (!is.integer(Number_of_Iterations) & !is.numeric(Number_of_Iterations))) {
    stop ("The 'Number_of_Iterations' argument must be of class 'integer' or 'numeric' and it must be of length 1.")
  }
  if (length(Thinning_Rate) != 1 | (!is.integer(Thinning_Rate) & !is.numeric(Thinning_Rate))) {
    stop ("The 'Thinning_Rate' argument must be of class 'integer' or 'numeric' and it must be of length 1.")
  }
  if (length(Burn_in_Value) != 1 | (!is.integer(Burn_in_Value) & !is.numeric(Burn_in_Value))) {
    stop ("The 'Burn_in_Value' argument must be of class 'integer' or 'numeric' and it must be of length 1.")
  }
  if (length(Number_of_Chains) != 1 | (!is.integer(Number_of_Chains) & !is.numeric(Number_of_Chains))) {
    stop ("The 'Number_of_Chains' argument must be of class 'integer' or 'numeric' and it must be of length 1.")
  }
  if (!is.character(Working_Directory) | !file.exists(Working_Directory)) {
    stop ("The 'Working_Directory' argument must be a character vector and it must be a valid path to a folder on this computer.")
  }
  
  
  # Calculating the Total Sum of Squares
  
  Total_Sum_of_Squares <- sum((Response - mean(Response)) ^ 2)
  
  
  # Generating a Logistic Function Model
  
  # Response = (1 / (1 + exp(-(Intercept + (Slope * Predictor)))))
  
  cat("\nLogistic Function Model\n\n")
  Logistic_Function_Model_Name <- "Logistic Function"
  Lowercase_Logistic_Function_Model_Name <- "logistic function"
  sink("Logistic Function Model.txt")
  cat("model {
    
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- (1 / (1 + exp(-(Intercept + (Slope * Predictor[i])))))
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Bayesian_p_Value")
  Logistic_Function_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "Logistic Function Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_Logistic_Function_Model_Response_Values <- (1 / (1 + exp(-(as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values)))))
  Logistic_Function_Model_Residual_Sum_of_Squares <- sum((Response - (1 / (1 + exp(-(as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor)))))) ^ 2)
  Logistic_Function_Model_Pseudo_R_Squared <- 1 - (Logistic_Function_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  Logistic_Function_Model <- paste0(Response_Name, " = (1 / (1 + exp(-(", as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")))))")
  Logistic_Function_Model_Bayesian_p_Value <- as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Logistic_Function_Model_Information <- list(Model_Name = Logistic_Function_Model_Name, Lowercase_Model_Name = Lowercase_Logistic_Function_Model_Name, Model = Logistic_Function_Model, Residual_Sum_of_Squares = Logistic_Function_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = Logistic_Function_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_Logistic_Function_Model_Response_Values, Output = Logistic_Function_Model_Output, Bayesian_p_Value = Logistic_Function_Model_Bayesian_p_Value)
  
  
  # Generating a Hyperbolic Tangent Model
  
  # Response = ((0.5 * tanh(Intercept + (Slope * Predictor))) + 0.5)
  
  cat("\n\nHyperbolic Tangent Model\n\n")
  Hyperbolic_Tangent_Model_Name <- "Hyperbolic Tangent"
  Lowercase_Hyperbolic_Tangent_Model_Name <- "hyperbolic tangent"
  sink("Hyperbolic Tangent Model.txt")
  cat("model {
    
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- ((0.5 * tanh(Intercept + (Slope * Predictor[i]))) + 0.5)
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Bayesian_p_Value")
  Hyperbolic_Tangent_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "Hyperbolic Tangent Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_Hyperbolic_Tangent_Model_Response_Values <- ((0.5 * tanh(as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values)) + 0.5)
  Hyperbolic_Tangent_Model_Residual_Sum_of_Squares <- sum((Response - ((0.5 * tanh(as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Slope) * Predictor)) + 0.5)) ^ 2)
  Hyperbolic_Tangent_Model_Pseudo_R_Squared <- 1 - (Hyperbolic_Tangent_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  Hyperbolic_Tangent_Model <- paste0(Response_Name, " = ((0.5 * tanh(", as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, "))) + 0.5)")
  Hyperbolic_Tangent_Model_Bayesian_p_Value <- as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Hyperbolic_Tangent_Model_Information <- list(Model_Name = Hyperbolic_Tangent_Model_Name, Lowercase_Model_Name = Lowercase_Hyperbolic_Tangent_Model_Name, Model = Hyperbolic_Tangent_Model, Residual_Sum_of_Squares = Hyperbolic_Tangent_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = Hyperbolic_Tangent_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_Hyperbolic_Tangent_Model_Response_Values, Output = Hyperbolic_Tangent_Model_Output, Bayesian_p_Value = Hyperbolic_Tangent_Model_Bayesian_p_Value)
  
  
  # Generating an Arctangent Function Model
  
  # Response = ((0.5 * ((2 / pi) * atan((pi / 2) * (Intercept + (Slope * Predictor))))) + 0.5)
  
  cat("\n\nArctangent Function Model\n\n")
  Arctangent_Function_Model_Name <- "Arctangent Function"
  Lowercase_Arctangent_Function_Model_Name <- "arctangent function"
  sink("Arctangent Function Model.txt")
  cat("model {
  
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    pi <- 3.14159265359
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- ((0.5 * ((2 / pi) * atan((pi / 2) * (Intercept + (Slope * Predictor[i]))))) + 0.5)
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Bayesian_p_Value")
  Arctangent_Function_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "Arctangent Function Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_Arctangent_Function_Model_Response_Values <- ((0.5 * ((2 / pi) * atan((pi / 2) * (as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values)))) + 0.5)
  Arctangent_Function_Model_Residual_Sum_of_Squares <- sum((Response - ((0.5 * ((2 / pi) * atan((pi / 2) * (as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor)))) + 0.5)) ^ 2)
  Arctangent_Function_Model_Pseudo_R_Squared <- 1 - (Arctangent_Function_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  Arctangent_Function_Model <- paste0(Response_Name, " = ((0.5 * ((2 / pi) * atan((pi / 2) * (", as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, "))))) + 0.5)")
  Arctangent_Function_Model_Bayesian_p_Value <- as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Arctangent_Function_Model_Information <- list(Model_Name = Arctangent_Function_Model_Name, Lowercase_Model_Name = Lowercase_Arctangent_Function_Model_Name, Model = Arctangent_Function_Model, Residual_Sum_of_Squares = Arctangent_Function_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = Arctangent_Function_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_Arctangent_Function_Model_Response_Values, Output = Arctangent_Function_Model_Output, Bayesian_p_Value = Arctangent_Function_Model_Bayesian_p_Value)
  
  
  # Generating a Gudermannian Function Model
  
  # Response = ((2 / pi) * atan(tanh((Intercept + (Slope * Predictor)) * pi / 4)) + 0.5)
  
  cat("\n\nGudermannian Function Model\n\n")
  Gudermannian_Function_Model_Name <- "Gudermannian Function"
  Lowercase_Gudermannian_Function_Model_Name <- "Gudermannian function"
  sink("Gudermannian Function Model.txt")
  cat("model {
    
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    pi <- 3.14159265359
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- ((2 / pi) * atan(tanh((Intercept + (Slope * Predictor[i])) * pi / 4)) + 0.5)
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Bayesian_p_Value")
  Gudermannian_Function_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "Gudermannian Function Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_Gudermannian_Function_Model_Response_Values <- ((2 / pi) * atan(tanh((as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values) * pi / 4)) + 0.5)
  Gudermannian_Function_Model_Residual_Sum_of_Squares <- sum((Response - ((2 / pi) * atan(tanh((as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) * pi / 4)) + 0.5)) ^ 2)
  Gudermannian_Function_Model_Pseudo_R_Squared <- 1 - (Gudermannian_Function_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  Gudermannian_Function_Model <- paste0(Response_Name, " = ((2 / pi) * atan(tanh((", as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) * pi / 4)) + 0.5)")
  Gudermannian_Function_Model_Bayesian_p_Value <- as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Gudermannian_Function_Model_Information <- list(Model_Name = Gudermannian_Function_Model_Name, Lowercase_Model_Name = Lowercase_Gudermannian_Function_Model_Name, Model = Gudermannian_Function_Model, Residual_Sum_of_Squares = Gudermannian_Function_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = Gudermannian_Function_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_Gudermannian_Function_Model_Response_Values, Output = Gudermannian_Function_Model_Output, Bayesian_p_Value = Gudermannian_Function_Model_Bayesian_p_Value)
  
  
  # Generating an Error Function Model
  
  # Response = ((0.5 * ((2 * pnorm((Intercept + (Slope * Predictor)) * sqrt(2), 0, 1)) - 1)) + 0.5)
  
  cat("\n\nError Function Model\n\n")
  Error_Function_Model_Name <- "Error Function"
  Lowercase_Error_Function_Model_Name <- "error function"
  sink("Error Function Model.txt")
  cat("model {
    
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    pi <- 3.14159265359
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- ((0.5 * ((2 * pnorm((Intercept + (Slope * Predictor[i])) * sqrt(2), 0, 1)) - 1)) + 0.5)
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Bayesian_p_Value")
  Error_Function_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "Error Function Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_Error_Function_Model_Response_Values <- ((0.5 * ((2 * pnorm((as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values)) * sqrt(2))) - 1)) + 0.5)
  Error_Function_Model_Residual_Sum_of_Squares <- sum((Response - ((0.5 * ((2 * pnorm((as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor)) * sqrt(2))) - 1)) + 0.5)) ^ 2)
  Error_Function_Model_Pseudo_R_Squared <- 1 - (Error_Function_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  Error_Function_Model <- paste0(Response_Name, " = ((0.5 * ((2 * pnorm((", as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Intercept)," + (", as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) * sqrt(2))) - 1)) + 0.5)")
  Error_Function_Model_Bayesian_p_Value <- as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Error_Function_Model_Information <- list(Model_Name = Error_Function_Model_Name, Lowercase_Model_Name = Lowercase_Error_Function_Model_Name, Model = Error_Function_Model, Residual_Sum_of_Squares = Error_Function_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = Error_Function_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_Error_Function_Model_Response_Values, Output = Error_Function_Model_Output, Bayesian_p_Value = Error_Function_Model_Bayesian_p_Value)
  
  
  # Generating a Generalised Logistic Function Model
  
  # Response = ((1 + exp(-(Intercept + (Slope * Predictor)))) ^ (-Exponent))
  
  cat("\n\nGeneralised Logistic Function Model\n\n")
  Generalised_Logistic_Function_Model_Name <- "Generalised Logistic Function"
  Lowercase_Generalised_Logistic_Function_Model_Name <- "generalised logistic function"
  sink("Generalised Logistic Function Model.txt")
  cat("model {
    
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Exponent ~ dlnorm(0, 1)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- ((1 + exp(-(Intercept + (Slope * Predictor[i])))) ^ (-Exponent))
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Exponent", "Bayesian_p_Value")
  Generalised_Logistic_Function_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "Generalised Logistic Function Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_Generalised_Logistic_Function_Model_Response_Values <- ((1 + exp(-(as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values))) ^ (-as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Exponent)))
  Generalised_Logistic_Function_Model_Residual_Sum_of_Squares <- sum((Response - ((1 + exp(-(as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor))) ^ (-as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Exponent)))) ^ 2)
  Generalised_Logistic_Function_Model_Pseudo_R_Squared <- 1 - (Generalised_Logistic_Function_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  Generalised_Logistic_Function_Model <- paste0(Response_Name, " = ((1 + exp(-(", as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")))) ^ (-", as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Exponent), "))")
  Generalised_Logistic_Function_Model_Bayesian_p_Value <- as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Generalised_Logistic_Function_Model_Information <- list(Model_Name = Generalised_Logistic_Function_Model_Name, Lowercase_Model_Name = Lowercase_Generalised_Logistic_Function_Model_Name, Model = Generalised_Logistic_Function_Model, Residual_Sum_of_Squares = Generalised_Logistic_Function_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = Generalised_Logistic_Function_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_Generalised_Logistic_Function_Model_Response_Values, Output = Generalised_Logistic_Function_Model_Output, Bayesian_p_Value = Generalised_Logistic_Function_Model_Bayesian_p_Value)
  
  
  # Generating an Algebraic Function Model
  
  # Response = ((0.5 * ((Intercept + (Slope * Predictor)) / sqrt(1 + ((Intercept + (Slope * Predictor)) ^ 2)))) + 0.5)
  
  cat("\n\nAlgebraic Function Model\n\n")
  Algebraic_Function_Model_Name <- "Algebraic Function"
  Lowercase_Algebraic_Function_Model_Name <- "algebraic function"
  sink("Algebraic Function Model.txt")
  cat("model {
    
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- ((0.5 * ((Intercept + (Slope * Predictor[i])) / sqrt(1 + ((Intercept + (Slope * Predictor[i])) ^ 2)))) + 0.5)
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Bayesian_p_Value")
  Algebraic_Function_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "Algebraic Function Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_Algebraic_Function_Model_Response_Values <- ((0.5 * ((as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values) / sqrt(1 + ((as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values) ^ 2)))) + 0.5)
  Algebraic_Function_Model_Residual_Sum_of_Squares <- sum((Response - ((0.5 * ((as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) / sqrt(1 + ((as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) ^ 2)))) + 0.5)) ^ 2)
  Algebraic_Function_Model_Pseudo_R_Squared <- 1 - (Algebraic_Function_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  Algebraic_Function_Model <- paste0(Response_Name, " = ((0.5 * ((", as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) / sqrt(1 + ((", as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) ^ 2)))) + 0.5)")
  Algebraic_Function_Model_Bayesian_p_Value <- as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Algebraic_Function_Model_Information <- list(Model_Name = Algebraic_Function_Model_Name, Lowercase_Model_Name = Lowercase_Algebraic_Function_Model_Name, Model = Algebraic_Function_Model, Residual_Sum_of_Squares = Algebraic_Function_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = Algebraic_Function_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_Algebraic_Function_Model_Response_Values, Output = Algebraic_Function_Model_Output, Bayesian_p_Value = Algebraic_Function_Model_Bayesian_p_Value)
  
  
  # Generating a More General Algebraic Function Model
  
  # Response = ((0.5 * ((Intercept + (Slope * Predictor)) / ((1 + (abs(Intercept + (Slope * Predictor)) ^ Exponent)) ^ (1 / Exponent)))) + 0.5)
  
  cat("\n\nA More General Algebraic Function Model\n\n")
  A_More_General_Algebraic_Function_Model_Name <- "A More General Algebraic Function"
  Lowercase_A_More_General_Algebraic_Function_Model_Name <- "a more general algebraic function"
  sink("A More General Algebraic Function Model.txt")
  cat("model {
  
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Exponent ~ dlnorm(0, 1)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- ((0.5 * ((Intercept + (Slope * Predictor[i])) / ((1 + (abs(Intercept + (Slope * Predictor[i])) ^ Exponent)) ^ (1 / Exponent)))) + 0.5)
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Exponent", "Bayesian_p_Value")
  A_More_General_Algebraic_Function_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "A More General Algebraic Function Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_A_More_General_Algebraic_Function_Model_Response_Values <- ((0.5 * ((as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values) / ((1 + (abs(as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values) ^ as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent))) ^ (1 / as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent))))) + 0.5)
  A_More_General_Algebraic_Function_Model_Residual_Sum_of_Squares <- sum((Response - ((0.5 * ((as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) / ((1 + (abs(as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) ^ as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent))) ^ (1 / as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent))))) + 0.5)) ^ 2)
  A_More_General_Algebraic_Logistic_Function_Model_Pseudo_R_Squared <- 1 - (A_More_General_Algebraic_Function_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  A_More_General_Algebraic_Function_Model <- paste0(Response_Name, " = ((0.5 * ((", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) / ((1 + (abs(", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) ^ ", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent), ")) ^ (1 / ", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent), ")))) + 0.5)")
  A_More_General_Algebraic_Function_Model_Bayesian_p_Value <- as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  A_More_General_Algebraic_Function_Model_Information <- list(Model_Name = A_More_General_Algebraic_Function_Model_Name, Lowercase_Model_Name = Lowercase_A_More_General_Algebraic_Function_Model_Name, Model = A_More_General_Algebraic_Function_Model, Residual_Sum_of_Squares = A_More_General_Algebraic_Function_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = A_More_General_Algebraic_Logistic_Function_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_A_More_General_Algebraic_Function_Model_Response_Values, Output = A_More_General_Algebraic_Function_Model_Output, Bayesian_p_Value = A_More_General_Algebraic_Function_Model_Bayesian_p_Value)
  
  
  # Generating a Gompertz Function Model
  
  # Response = (exp(-exp(Intercept + (Slope * Predictor))))
  
  cat("\n\nGompertz Function Model\n\n")
  Gompertz_Function_Model_Name <- "Gompertz Function"
  Lowercase_Gompertz_Function_Model_Name <- "Gompertz function"
  sink("Gompertz Function Model.txt")
  cat("model {
    
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- (exp(-exp(Intercept + (Slope * Predictor[i]))))
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Bayesian_p_Value")
  Gompertz_Function_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "Gompertz Function Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_Gompertz_Function_Model_Response_Values <- (exp(-exp(as.numeric(Gompertz_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Gompertz_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values))))
  Gompertz_Function_Model_Residual_Sum_of_Squares <- sum((Response - (exp(-exp(as.numeric(Gompertz_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Gompertz_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor))))) ^ 2)
  Gompertz_Function_Model_Pseudo_R_Squared <- 1 - (Gompertz_Function_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  Gompertz_Function_Model <- paste0(Response_Name, " = (exp(-exp(", as.numeric(Gompertz_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Gompertz_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, "))))")
  Gompertz_Function_Model_Bayesian_p_Value <- as.numeric(Gompertz_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Gompertz_Function_Model_Information <- list(Model_Name = Gompertz_Function_Model_Name, Lowercase_Model_Name = Lowercase_Gompertz_Function_Model_Name, Model = Gompertz_Function_Model, Residual_Sum_of_Squares = Gompertz_Function_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = Gompertz_Function_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_Gompertz_Function_Model_Response_Values, Output = Gompertz_Function_Model_Output, Bayesian_p_Value = Gompertz_Function_Model_Bayesian_p_Value)
  
  
  # Generating a Gompertz Function Model That Has Been Rotated by 180 Degrees
  
  # Response = (1 - (exp(-exp(Intercept + (Slope * Predictor)))))
  
  cat("\n\nRotated Gompertz Function Model\n\n")
  Rotated_Gompertz_Function_Model_Name <- "Rotated Gompertz Function"
  Lowercase_Rotated_Gompertz_Function_Model_Name <- "rotated Gompertz function"
  sink("Rotated Gompertz Function Model.txt")
  cat("model {
    
    # Priors
    Intercept ~ dnorm(0, 0.001)
    Slope ~ dnorm(0, 0.001)
    Sigma ~ dlnorm(0, 1)
    Tau <- (1 / (Sigma ^ 2))
    
    # Likelihood and Model Fit
    for (i in 1:Number_of_Observations) {
      Response[i] ~ dnorm(Mean[i], Tau)
      Mean[i] <- (1 - (exp(-exp(Intercept + (Slope * Predictor[i])))))
      Actual_Squared_Residual[i] <- ((Response[i] - Mean[i]) ^ 2)
      Simulated_Response[i] ~ dbern(Mean[i])
      Simulated_Squared_Residual[i] <- ((Simulated_Response[i] - Mean[i]) ^ 2)
    }
    Bayesian_p_Value <- step((sum((Simulated_Squared_Residual[]) ^ 2)) / (sum((Actual_Squared_Residual[]) ^ 2)) - 1) 
    
  }", fill = T)
  sink()
  Parameters <- c("Intercept", "Slope", "Bayesian_p_Value")
  Rotated_Gompertz_Function_Model_Output <- R2jags::jags(Data, Initial_Values, Parameters, "Rotated Gompertz Function Model.txt", n.chains = Number_of_Chains, n.thin = Thinning_Rate, n.iter = Number_of_Iterations, n.burnin = Burn_in_Value, working.directory = Working_Directory)
  Fitted_Rotated_Gompertz_Function_Model_Response_Values <- (1 - (exp(-exp(as.numeric(Rotated_Gompertz_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Rotated_Gompertz_Function_Model_Output$BUGSoutput$mean$Slope) * Fitted_Predictor_Values)))))
  Rotated_Gompertz_Function_Model_Residual_Sum_of_Squares <- sum((Response - (1 - (exp(-exp(as.numeric(Rotated_Gompertz_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Rotated_Gompertz_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor)))))) ^ 2)
  Rotated_Gompertz_Function_Model_Pseudo_R_Squared <- 1 - (Rotated_Gompertz_Function_Model_Residual_Sum_of_Squares / Total_Sum_of_Squares)
  Rotated_Gompertz_Function_Model <- paste0(Response_Name, " = (1 - (exp(-exp(", as.numeric(Rotated_Gompertz_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Rotated_Gompertz_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")))))")
  Rotated_Gompertz_Function_Model_Bayesian_p_Value <- as.numeric(Rotated_Gompertz_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Rotated_Gompertz_Function_Model_Information <- list(Model_Name = Rotated_Gompertz_Function_Model_Name, Lowercase_Model_Name = Lowercase_Rotated_Gompertz_Function_Model_Name, Model = Rotated_Gompertz_Function_Model, Residual_Sum_of_Squares = Rotated_Gompertz_Function_Model_Residual_Sum_of_Squares, Pseudo_R_Squared = Rotated_Gompertz_Function_Model_Pseudo_R_Squared, Fitted_Response_Values = Fitted_Rotated_Gompertz_Function_Model_Response_Values, Output = Rotated_Gompertz_Function_Model_Output, Bayesian_p_Value = Rotated_Gompertz_Function_Model_Bayesian_p_Value)
  
  
  # Compiling the Models Into One List
  
  Model_List <- list(Logistic_Function_Model = Logistic_Function_Model_Information, Hyperbolic_Tangent_Model = Hyperbolic_Tangent_Model_Information, Arctangent_Function_Model = Arctangent_Function_Model_Information, Gudermannian_Function_Model = Gudermannian_Function_Model_Information, Error_Function_Model_Information = Error_Function_Model_Information, Generalised_Logistic_Function_Model = Generalised_Logistic_Function_Model_Information, Algebraic_Function_Model = Algebraic_Function_Model_Information, A_More_General_Algebraic_Function_Model = A_More_General_Algebraic_Function_Model_Information, Gompertz_Function_Model = Gompertz_Function_Model_Information, Rotated_Gompertz_Function_Model = Rotated_Gompertz_Function_Model_Information)
  
  
  # Returning the Pertinent Model Information
  
  Pertinent_Model_Information_List <- lapply(Model_List, function (x) {
    list(Model_Name = x$Model_Name, Model = x$Model, Fitted_Values = data.frame(Predictor = Fitted_Predictor_Values, Response = x$Fitted_Response_Values), Residual_Sum_of_Squares = x$Residual_Sum_of_Squares, Pseudo_R_Squared = x$Pseudo_R_Squared, Output = x$Output$BUGSoutput$summary)
  })
  Pertinent_Model_Information_List <- list(Model_Information = Pertinent_Model_Information_List, Conclusion = paste0("The model that best fits the data is the ", unlist(sapply(Model_List, `[`, 'Lowercase_Model_Name'))[which.max(unlist(sapply(Model_List, `[`, 'Pseudo_R_Squared')))], " model."))
  class(Pertinent_Model_Information_List) <- "Custom_Class"
  cat("\n\nFunction Output:\n\n")
  return (Pertinent_Model_Information_List)
  
}

print.Custom_Class <- function (x) {
  print(c(lapply(Function_Output$Model_Information, `[`, c('Model_Name', 'Model', 'Residual_Sum_of_Squares', 'Pseudo_R_Squared', 'Output')), Conclusion = x$Conclusion))
}


# An Example

# Generating Some Practice Data

Number_of_Observations <- 100
Minimum_Predictor_Value <- 0
Maximum_Predictor_Value <- 25
Predictor_Variable <- runif(Number_of_Observations, Minimum_Predictor_Value, Maximum_Predictor_Value)
Response_Variable <- rbinom(Number_of_Observations, 1, (Predictor_Variable - Minimum_Predictor_Value) / (Maximum_Predictor_Value - Minimum_Predictor_Value))
Data_Frame <- data.frame(Predictor_Variable = Predictor_Variable, Response_Variable = Response_Variable)

# Test the Function Out

(Function_Output <- Function_for_Fitting_an_Optimal_Sigmoid_Model(Predictor_Variable, Response_Variable, Data_Frame))

# Here's the output from the preceding line of code.

# > (Function_Output <- Function_for_Fitting_an_Optimal_Sigmoid_Model(Predictor_Variable, Response_Variable, Data_Frame))
# 
# Logistic Function Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 103
# Total graph size: 1319
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# Hyperbolic Tangent Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 103
# Total graph size: 1220
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# Arctangent Function Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 103
# Total graph size: 1423
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# Gudermannian Function Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 103
# Total graph size: 1523
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# Error Function Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 103
# Total graph size: 1522
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# Generalised Logistic Function Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 104
# Total graph size: 1321
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# Algebraic Function Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 103
# Total graph size: 1520
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# A More General Algebraic Function Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 104
# Total graph size: 1622
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# Gompertz Function Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 103
# Total graph size: 1219
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# Rotated Gompertz Function Model
# 
# Compiling model graph
# Resolving undeclared variables
# Allocating nodes
# Graph information:
#   Observed stochastic nodes: 100
# Unobserved stochastic nodes: 103
# Total graph size: 1319
# 
# Initializing model
# 
# |++++++++++++++++++++++++++++++++++++++++++++++++++| 100%
# |**************************************************| 100%
# 
# 
# Function Output:
# 
# $Logistic_Function_Model
# $Logistic_Function_Model$Model_Name
# [1] "Logistic Function"
# 
# $Logistic_Function_Model$Model
# [1] "Response_Variable = (1 / (1 + exp(-(-2.33644903978487 + (0.186481072480213 * Predictor_Variable)))))"
# 
# $Logistic_Function_Model$Residual_Sum_of_Squares
# [1] 18.85
# 
# $Logistic_Function_Model$Pseudo_R_Squared
# [1] 0.2456983
# 
# $Logistic_Function_Model$Output
#                         mean         sd       2.5%         25%         50%         75%       97.5%     Rhat  n.eff
# Bayesian_p_Value   0.4396061 0.49634001   0.000000   0.0000000   0.0000000   1.0000000   1.0000000 1.000996 300000
# Intercept         -2.3364490 0.65441211  -3.813642  -2.7056482  -2.2661864  -1.8854810  -1.2684210 1.001025  64000
# Slope              0.1864811 0.04973225   0.106909   0.1522776   0.1807865   0.2138063   0.2999475 1.001004 200000
# deviance         120.0063385 2.74402837 116.918649 118.0225725 119.2915799 121.2159109 127.1293145 1.001121  16000
# 
# 
# $Hyperbolic_Tangent_Model
# $Hyperbolic_Tangent_Model$Model_Name
# [1] "Hyperbolic Tangent"
# 
# $Hyperbolic_Tangent_Model$Model
# [1] "Response_Variable = ((0.5 * tanh(-1.16798978417337 + (0.0931924756513314 * Predictor_Variable))) + 0.5)"
# 
# $Hyperbolic_Tangent_Model$Residual_Sum_of_Squares
# [1] 18.84958
# 
# $Hyperbolic_Tangent_Model$Pseudo_R_Squared
# [1] 0.2457152
# 
# $Hyperbolic_Tangent_Model$Output
#                          mean         sd         2.5%          25%          50%         75%       97.5%     Rhat n.eff
# Bayesian_p_Value   0.43967340 0.49634820   0.00000000   0.00000000   0.00000000   1.0000000   1.0000000 1.001110 17000
# Intercept         -1.16798978 0.32656904  -1.91258496  -1.35089897  -1.13239786  -0.9441194  -0.6302550 1.001457  4300
# Slope              0.09319248 0.02478972   0.05318255   0.07623067   0.09030607   0.1067146   0.1500693 1.001377  5200
# deviance         120.00271230 2.74062447 116.91427815 118.01306229 119.28504546 121.2293770 127.0787605 1.001111 17000
# 
# 
# $Arctangent_Function_Model
# $Arctangent_Function_Model$Model_Name
# [1] "Arctangent Function"
# 
# $Arctangent_Function_Model$Model
# [1] "Response_Variable = ((0.5 * ((2 / pi) * atan((pi / 2) * (-1.51767318209146 + (0.123473532957025 * Predictor_Variable))))) + 0.5)"
# 
# $Arctangent_Function_Model$Residual_Sum_of_Squares
# [1] 19.13632
# 
# $Arctangent_Function_Model$Pseudo_R_Squared
# [1] 0.2342411
# 
# $Arctangent_Function_Model$Output
#                         mean         sd         2.5%          25%         50%         75%       97.5%     Rhat  n.eff
# Bayesian_p_Value   0.4296936 0.49503317   0.00000000   0.00000000   0.0000000   1.0000000   1.0000000 1.000994 300000
# Intercept         -1.5176732 0.57344177  -2.90463611  -1.78850939  -1.4207121  -1.1301310  -0.7090572 1.000998 300000
# Slope              0.1234735 0.04668362   0.05968212   0.09216846   0.1150764   0.1447289   0.2367473 1.000994 300000
# deviance         121.3204955 2.97635880 117.98505996 119.16693135 120.5409938 122.6270708 129.1343053 1.001046  38000
# 
# 
# $Gudermannian_Function_Model
# $Gudermannian_Function_Model$Model_Name
# [1] "Gudermannian Function"
# 
# $Gudermannian_Function_Model$Model
# [1] "Response_Variable = ((2 / pi) * atan(tanh((-1.20805893114019 + (0.0966429878324961 * Predictor_Variable)) * pi / 4)) + 0.5)"
# 
# $Gudermannian_Function_Model$Residual_Sum_of_Squares
# [1] 18.88393
# 
# $Gudermannian_Function_Model$Pseudo_R_Squared
# [1] 0.2443404
# 
# $Gudermannian_Function_Model$Output
#                          mean         sd         2.5%          25%          50%         75%       97.5%     Rhat n.eff
# Bayesian_p_Value   0.43988215 0.49637352   0.00000000   0.00000000   0.00000000   1.0000000   1.0000000 1.001053 33000
# Intercept         -1.20805893 0.34776920  -2.00090330  -1.40100863  -1.16870193  -0.9696084  -0.6451038 1.001270  7200
# Slope              0.09664299 0.02664765   0.05434613   0.07839391   0.09340856   0.1110425   0.1580554 1.001196  9900
# deviance         120.14978269 2.76001653 117.06630765 118.15899834 119.42265263 121.3555495 127.3595701 1.001278  7000
# 
# 
# $Error_Function_Model_Information
# $Error_Function_Model_Information$Model_Name
# [1] "Error Function"
# 
# $Error_Function_Model_Information$Model
# [1] "Response_Variable = ((0.5 * ((2 * pnorm((-0.991378525508604 + (0.0789711310566494 * Predictor_Variable)) * sqrt(2))) - 1)) + 0.5)"
# 
# $Error_Function_Model_Information$Residual_Sum_of_Squares
# [1] 18.80415
# 
# $Error_Function_Model_Information$Pseudo_R_Squared
# [1] 0.2475332
# 
# $Error_Function_Model_Information$Output
#                          mean         sd         2.5%          25%          50%          75%       97.5%     Rhat  n.eff
# Bayesian_p_Value   0.44330640 0.49677627   0.00000000   0.00000000   0.00000000   1.00000000   1.0000000 1.000997 300000
# Intercept         -0.99137853 0.26540684  -1.58289988  -1.14226200  -0.96597784  -0.80892560  -0.5498642 1.000995 300000
# Slope              0.07897113 0.01992914   0.04633745   0.06535553   0.07686405   0.09015762   0.1236903 1.001003 200000
# deviance         119.79883423 2.72879377 116.73696381 117.82682949 119.08605745 121.00649886 126.8836311 1.001063  29000
# 
# 
# $Generalised_Logistic_Function_Model
# $Generalised_Logistic_Function_Model$Model_Name
# [1] "Generalised Logistic Function"
# 
# $Generalised_Logistic_Function_Model$Model
# [1] "Response_Variable = ((1 + exp(-(-8.32268644149368 + (0.415027012251236 * Predictor_Variable)))) ^ (-0.776179983894283))"
# 
# $Generalised_Logistic_Function_Model$Residual_Sum_of_Squares
# [1] 27.04924
# 
# $Generalised_Logistic_Function_Model$Pseudo_R_Squared
# [1] -0.08240268
# 
# $Generalised_Logistic_Function_Model$Output
#                         mean        sd         2.5%         25%         50%         75%       97.5%     Rhat n.eff
# Bayesian_p_Value   0.4532727 0.4978126   0.00000000   0.0000000   0.0000000   1.0000000   1.0000000 1.001096 19000
# Exponent           0.7761800 1.3599120   0.05865472   0.1637466   0.3666564   0.8459431   3.8686341 1.006814   350
# Intercept         -8.3226864 8.1000283 -30.45132278 -11.5134493  -5.5476127  -2.6700360   0.0374854 1.016070   250
# Slope              0.4150270 0.3303920   0.11180621   0.1915711   0.2920673   0.5243842   1.3433115 1.008908   280
# deviance         120.2631854 2.7646594 117.11119765 118.2728544 119.5540138 121.4862702 127.4287175 1.001192 10000
# 
# 
# $Algebraic_Function_Model
# $Algebraic_Function_Model$Model_Name
# [1] "Algebraic Function"
# 
# $Algebraic_Function_Model$Model
# [1] "Response_Variable = ((0.5 * ((-1.28228164399336 + (0.102990152273513 * Predictor_Variable)) / sqrt(1 + ((-1.28228164399336 + (0.102990152273513 * Predictor_Variable)) ^ 2)))) + 0.5)"
# 
# $Algebraic_Function_Model$Residual_Sum_of_Squares
# [1] 18.9553
# 
# $Algebraic_Function_Model$Pseudo_R_Squared
# [1] 0.2414848
# 
# $Algebraic_Function_Model$Output
#                         mean        sd         2.5%          25%          50%         75%       97.5%     Rhat  n.eff
# Bayesian_p_Value   0.4381886 0.4961655   0.00000000   0.00000000   0.00000000   1.0000000   1.0000000 1.000998 300000
# Intercept         -1.2822816 0.3985051  -2.21367787  -1.49430741  -1.22923782  -1.0091010  -0.6616935 1.001132  14000
# Slope              0.1029902 0.0312075   0.05563512   0.08159677   0.09849964   0.1191179   0.1764576 1.001083  22000
# deviance         120.4998280 2.8101698 117.33391496 118.45750900 119.76532190 121.7547884 127.7964876 1.001149  13000
# 
# 
# $A_More_General_Algebraic_Function_Model
# $A_More_General_Algebraic_Function_Model$Model_Name
# [1] "A More General Algebraic Function"
# 
# $A_More_General_Algebraic_Function_Model$Model
# [1] "Response_Variable = ((0.5 * ((-33.1017585392326 + (3.17187376338609 * Predictor_Variable)) / ((1 + (abs(-33.1017585392326 + (3.17187376338609 * Predictor_Variable)) ^ 0.452356234695712)) ^ (1 / 0.452356234695712)))) + 0.5)"
# 
# $A_More_General_Algebraic_Function_Model$Residual_Sum_of_Squares
# [1] 20.32901
# 
# $A_More_General_Algebraic_Function_Model$Pseudo_R_Squared
# [1] 0.1865141
# 
# $A_More_General_Algebraic_Function_Model$Output
#                         mean         sd        2.5%        25%         50%         75%       97.5%     Rhat  n.eff
# Bayesian_p_Value   0.5063872  0.4999600   0.0000000   0.000000   1.0000000   1.0000000   1.0000000 1.001030  54000
# Exponent           0.4523562  0.3423357   0.2793075   0.348172   0.3944637   0.4587175   0.9507556 1.011761   1600
# Intercept        -33.1017585 20.6821724 -79.6188051 -45.994569 -30.3923948 -17.2816329  -2.2224198 1.001913   2200
# Slope              3.1718738  2.1443182   0.1884297   1.557175   2.8196605   4.3954504   8.1507183 1.002959   1800
# deviance         124.1738211  2.6201282 120.8163535 122.485443 123.5674833 125.2809483 130.7915774 1.001727 300000
# 
# 
# $Gompertz_Function_Model
# $Gompertz_Function_Model$Model_Name
# [1] "Gompertz Function"
# 
# $Gompertz_Function_Model$Model
# [1] "Response_Variable = (exp(-exp(1.17911288434877 + (-0.130132076554229 * Predictor_Variable))))"
# 
# $Gompertz_Function_Model$Residual_Sum_of_Squares
# [1] 18.82781
# 
# $Gompertz_Function_Model$Pseudo_R_Squared
# [1] 0.2465861
# 
# $Gompertz_Function_Model$Output
#                         mean         sd        2.5%         25%         50%         75%       97.5%     Rhat  n.eff
# Bayesian_p_Value   0.4129293 0.49236116   0.0000000   0.0000000   0.0000000   1.0000000   1.0000000 1.001030  55000
# Intercept          1.1791129 0.41164251   0.5046325   0.9000821   1.1342711   1.4054446   2.1154189 1.001025  64000
# Slope             -0.1301321 0.03439938  -0.2089268  -0.1488089  -0.1260977  -0.1067671  -0.0748665 1.001023  67000
# deviance         119.8812905 2.73266075 116.8042605 117.8977374 119.1696986 121.0907041 127.0207659 1.000998 300000
# 
# 
# $Rotated_Gompertz_Function_Model
# $Rotated_Gompertz_Function_Model$Model_Name
# [1] "Rotated Gompertz Function"
# 
# $Rotated_Gompertz_Function_Model$Model
# [1] "Response_Variable = (1 - (exp(-exp(-2.04999535959946 + (0.12686703664502 * Predictor_Variable)))))"
# 
# $Rotated_Gompertz_Function_Model$Residual_Sum_of_Squares
# [1] 18.85588
# 
# $Rotated_Gompertz_Function_Model$Pseudo_R_Squared
# [1] 0.2454632
# 
# $Rotated_Gompertz_Function_Model$Output
#                         mean         sd         2.5%         25%         50%         75%       97.5%     Rhat n.eff
# Bayesian_p_Value   0.4527542 0.49776367   0.00000000   0.0000000   0.0000000   1.0000000   1.0000000 1.001017 87000
# Intercept         -2.0499954 0.47015846  -3.10114575  -2.3236138  -2.0040957  -1.7241739  -1.2609322 1.001087 21000
# Slope              0.1268670 0.03184876   0.07418693   0.1049918   0.1236174   0.1448527   0.1985734 1.001074 25000
# deviance         120.0744383 2.73859450 116.99208788 118.0875099 119.3619519 121.2891779 127.1699346 1.001061 30000
# 
# 
# $Conclusion
# [1] "The model that best fits the data is the error function model."


# Generating a Plot of All the Models

# Make sure to expand the plotting window.

Color <- rainbow(length(Function_Output$Model_Information))
par(mar = c(12, 4, 4, 2))
plot(Response_Variable ~ Predictor_Variable, Data_Frame, main = "Fitting Sigmoid Models to the Data", xlab = "", ylab = "", pch = 19, type = 'n')
title(xlab = "Predictor Variable", line = 2.5)
title(ylab = "Response Variable", line = 2.5)
for (i in seq_len(length(Function_Output$Model_Information))) {
  lines(Function_Output$Model_Information[[i]]$Fitted_Values$Response ~ Function_Output$Model_Information[[i]]$Fitted_Values$Predictor, col = Color[i], lwd = 2)
}
points(Response_Variable ~ Predictor_Variable, Data_Frame, pch = 19)
legend("bottom", xpd = T, ncol = 2, inset = c(0, -0.55), title = expression(paste("Model Type (Pseudo ", italic("R") ^ "2" * ")")), legend = paste0(unlist(sapply(Function_Output$Model_Information, `[`, 'Model_Name')), " (", format(round(unlist(sapply(Function_Output$Model_Information, `[`, 'Pseudo_R_Squared')), 3), nsmall = 3), ")"), col = Color, lty = 1, lwd = 2)


# Works Cited

# Su, Y.-S., and M. Yajima. 2021. R2jags: Using R to Run 'JAGS'. R package
# version 0.7-1. https://cran.r-project.org/web/packages/R2jags/.
