
# A Function for Determining the Optimal Sigmoid Function for Fitting Binary
# Response Data

# David Moore
# davidblakneymoore@gmail.com
# May 2022


# The Explanation

# Though logistic regression is almost exclusively used for modeling
# probabilities based on binary response data, there are many other options.
# This function fits eight separate models to binary response data and
# determines which of the eight are best based on the sum of squared residuals.
# The eight model types are based on the logistic function, the hyperbolic
# tangent, the arctangent function, the Gudermannian function, the error
# function, a generalised logistic function, an algebraic function, and a more
# general algebraic function. All eight of these functions have been rescaled
# so that they are bounded by 0 and 1 on the response variable axis.

# This function uses the 'R2jags' package heavily and it returns all the
# pertinent information for each model. For each model, it returns the model
# name, the actual model (with the parameters that best fit the data included),
# a data frame of fitted predictors and responses for plotting, the sum of the
# squared residuals, and the output from the Bayesian analysis, which includes
# all the coefficients and the Bayesian p value.

# Unfortunately, this function can't handle multiple predictor variables yet.
# Some day I may update it to be able to account for multiple predictor
# variables.

# I took an outstanding course in Bayesian statistics with Dr. Remington Moll
# at the University of New Hampshire that really helped me write this function.

# This function takes 8 arguments. The first two are required.

# 'Predictor' is a vector of predictor variables to be used in the model.

# 'Response' is a vector of response variables to be used in the model. It
# should be binary (1s and 0s).

# 'Data_Frame' is an optional data frame to include such that column names can
# be supplied for the 'Predictor' and the 'Response' arguments. The data frame
# that these columns are from should be provided for this Data_Frame argument.

# 'Number_of_Iterations = 1000' is the number of iterations each Markov chain
# Monte Carlo simulation undergoes. The default is '1000'. More iterations lead
# to more precise results, but the program will take longer to run.

# 'Thinning_Rate = 1' describes how many iterations of the Markov chain Monte
# Carlo simulation are performed per stored value. The default is '1'. For
# example, if the thinning rate is 3, every third iteration of the Markov chain
# Monte Carlo simulation would be stored as model output.

# 'Burn_in_Value = 100' is the number of initial iterations of each Markov
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

Function_for_Fitting_an_Optimal_Sigmoid_Model <- function (Predictor, Response, Data_Frame, Number_of_Iterations = 1000, Thinning_Rate = 1, Burn_in_Value = 100, Number_of_Chains = 3, Working_Directory = getwd()) {
  
  # Prepare the Data
  
  Predictor_Name <- gsub("^.*[$]", "", deparse(substitute(Predictor)))
  Response_Name <- gsub("^.*[$]", "", deparse(substitute(Response)))
  if (!missing(Data_Frame)) {
    if (class(Data_Frame) != 'data.frame') {
      stop ("'Data_Frame' must be of class 'data.frame'.")
    }
    Data_Frame <- Data_Frame[, c(Predictor_Name, Response_Name)]
  } else if (missing(Data_Frame)) {
    Data_Frame <- data.frame(Predictor, Response)
  }
  Predictor <- Data_Frame[, which(colnames(Data_Frame) == Predictor_Name)]
  Response <- Data_Frame[, which(colnames(Data_Frame) == Response_Name)]
  Data <- list(Predictor = Data_Frame[, which(colnames(Data_Frame) == Predictor_Name)], Response = Data_Frame[, which(colnames(Data_Frame) == Response_Name)], Number_of_Observations = nrow(Data_Frame))
  Initial_Values <- function () {
    list()
  }
  Fitted_Predictor_Values <- seq(Minimum_Predictor_Value, Maximum_Predictor_Value, length.out = 100000)
  
    
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
  
    
  # Generating a Logistic Function Model
  
  # Response = (1 / (1 + exp(-(Intercept + (Slope * Predictor)))))
  
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
  Logistic_Function_Model_Sum_of_Squared_Residuals <- sum((Response - (1 / (1 + exp(-(as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor)))))) ^ 2)
  Logistic_Function_Model <- paste0(Response_Name, " = (1 / (1 + exp(-(", as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")))))")
  Logistic_Function_Model_Bayesian_p_Value <- as.numeric(Logistic_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Logistic_Function_Model_Information <- list(Model_Name = Logistic_Function_Model_Name, Lowercase_Model_Name = Lowercase_Logistic_Function_Model_Name, Model = Logistic_Function_Model, Sum_of_Squared_Residuals = Logistic_Function_Model_Sum_of_Squared_Residuals, Fitted_Response_Values = Fitted_Logistic_Function_Model_Response_Values, Output = Logistic_Function_Model_Output, Bayesian_p_Value = Logistic_Function_Model_Bayesian_p_Value)
  
  
  # Generating a Hyperbolic Tangent Model
  
  # Response = ((0.5 * tanh(Intercept + (Slope * Predictor))) + 0.5)
  
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
  Hyperbolic_Tangent_Model_Sum_of_Squared_Residuals <- sum((Response - ((0.5 * tanh(as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Slope) * Predictor)) + 0.5)) ^ 2)
  Hyperbolic_Tangent_Model <- paste0(Response_Name, " = ((0.5 * tanh(", as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, "))) + 0.5)")
  Hyperbolic_Tangent_Model_Bayesian_p_Value <- as.numeric(Hyperbolic_Tangent_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Hyperbolic_Tangent_Model_Information <- list(Model_Name = Hyperbolic_Tangent_Model_Name, Lowercase_Model_Name = Lowercase_Hyperbolic_Tangent_Model_Name, Model = Hyperbolic_Tangent_Model, Sum_of_Squared_Residuals = Hyperbolic_Tangent_Model_Sum_of_Squared_Residuals, Fitted_Response_Values = Fitted_Hyperbolic_Tangent_Model_Response_Values, Output = Hyperbolic_Tangent_Model_Output, Bayesian_p_Value = Hyperbolic_Tangent_Model_Bayesian_p_Value)
  
  
  # Generating an Arctangent Function Model
  
  # Response = ((0.5 * ((2 / pi) * atan((pi / 2) * (Intercept + (Slope * Predictor))))) + 0.5)
  
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
  Arctangent_Function_Model_Sum_of_Squared_Residuals <- sum((Response - ((0.5 * ((2 / pi) * atan((pi / 2) * (as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor)))) + 0.5)) ^ 2)
  Arctangent_Function_Model <- paste0(Response_Name, " = ((0.5 * ((2 / pi) * atan((pi / 2) * (", as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, "))))) + 0.5)")
  Arctangent_Function_Model_Bayesian_p_Value <- as.numeric(Arctangent_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Arctangent_Function_Model_Information <- list(Model_Name = Arctangent_Function_Model_Name, Lowercase_Model_Name = Lowercase_Arctangent_Function_Model_Name, Model = Arctangent_Function_Model, Sum_of_Squared_Residuals = Arctangent_Function_Model_Sum_of_Squared_Residuals, Fitted_Response_Values = Fitted_Arctangent_Function_Model_Response_Values, Output = Arctangent_Function_Model_Output, Bayesian_p_Value = Arctangent_Function_Model_Bayesian_p_Value)
  
  
  # Generating a Gudermannian Function Model
  
  # Response = ((2 / pi) * atan(tanh((Intercept + (Slope * Predictor)) * pi / 4)) + 0.5)
  
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
  Gudermannian_Function_Model_Sum_of_Squared_Residuals <- sum((Response - ((2 / pi) * atan(tanh((as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) * pi / 4)) + 0.5)) ^ 2)
  Gudermannian_Function_Model <- paste0(Response_Name, " = ((2 / pi) * atan(tanh((", as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) * pi / 4)) + 0.5)")
  Gudermannian_Function_Model_Bayesian_p_Value <- as.numeric(Gudermannian_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Gudermannian_Function_Model_Information <- list(Model_Name = Gudermannian_Function_Model_Name, Lowercase_Model_Name = Lowercase_Gudermannian_Function_Model_Name, Model = Gudermannian_Function_Model, Sum_of_Squared_Residuals = Gudermannian_Function_Model_Sum_of_Squared_Residuals, Fitted_Response_Values = Fitted_Gudermannian_Function_Model_Response_Values, Output = Gudermannian_Function_Model_Output, Bayesian_p_Value = Gudermannian_Function_Model_Bayesian_p_Value)
  
  
  # Generating an Error Function Model
  
  # Response = ((0.5 * ((2 * pnorm((Intercept + (Slope * Predictor)) * sqrt(2), 0, 1)) - 1)) + 0.5)
  
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
  Error_Function_Model_Sum_of_Squared_Residuals <- sum((Response - ((0.5 * ((2 * pnorm((as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Intercept) + (as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor)) * sqrt(2))) - 1)) + 0.5)) ^ 2)
  Error_Function_Model <- paste0(Response_Name, " = ((0.5 * ((2 * pnorm((", as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Intercept)," + (", as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) * sqrt(2))) - 1)) + 0.5)")
  Error_Function_Model_Bayesian_p_Value <- as.numeric(Error_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Error_Function_Model_Information <- list(Model_Name = Error_Function_Model_Name, Lowercase_Model_Name = Lowercase_Error_Function_Model_Name, Model = Error_Function_Model, Sum_of_Squared_Residuals = Error_Function_Model_Sum_of_Squared_Residuals, Fitted_Response_Values = Fitted_Error_Function_Model_Response_Values, Output = Error_Function_Model_Output, Bayesian_p_Value = Error_Function_Model_Bayesian_p_Value)
  
  
  # Generating a Generalised Logistic Function Model
  
  # Response = ((1 + exp(-(Intercept + (Slope * Predictor)))) ^ (-Exponent))
  
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
  Generalised_Logistic_Function_Model_Sum_of_Squared_Residuals <- sum((Response - ((1 + exp(-(as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor))) ^ (-as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Exponent)))) ^ 2)
  Generalised_Logistic_Function_Model <- paste0(Response_Name, " = ((1 + exp(-(", as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")))) ^ (-", as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Exponent), "))")
  Generalised_Logistic_Function_Model_Bayesian_p_Value <- as.numeric(Generalised_Logistic_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Generalised_Logistic_Function_Model_Information <- list(Model_Name = Generalised_Logistic_Function_Model_Name, Lowercase_Model_Name = Lowercase_Generalised_Logistic_Function_Model_Name, Model = Generalised_Logistic_Function_Model, Sum_of_Squared_Residuals = Generalised_Logistic_Function_Model_Sum_of_Squared_Residuals, Fitted_Response_Values = Fitted_Generalised_Logistic_Function_Model_Response_Values, Output = Generalised_Logistic_Function_Model_Output, Bayesian_p_Value = Generalised_Logistic_Function_Model_Bayesian_p_Value)
  
  
  # Generating an Algebraic Function Model
  
  # Response = ((0.5 * ((Intercept + (Slope * Predictor)) / sqrt(1 + ((Intercept + (Slope * Predictor)) ^ 2)))) + 0.5)
  
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
  Algebraic_Function_Model_Sum_of_Squared_Residuals <- sum((Response - ((0.5 * ((as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) / sqrt(1 + ((as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) ^ 2)))) + 0.5)) ^ 2)
  Algebraic_Function_Model <- paste0(Response_Name, " = ((0.5 * ((", as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) / sqrt(1 + ((", as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) ^ 2)))) + 0.5)")
  Algebraic_Function_Model_Bayesian_p_Value <- as.numeric(Algebraic_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  Algebraic_Function_Model_Information <- list(Model_Name = Algebraic_Function_Model_Name, Lowercase_Model_Name = Lowercase_Algebraic_Function_Model_Name, Model = Algebraic_Function_Model, Sum_of_Squared_Residuals = Algebraic_Function_Model_Sum_of_Squared_Residuals, Fitted_Response_Values = Fitted_Algebraic_Function_Model_Response_Values, Output = Algebraic_Function_Model_Output, Bayesian_p_Value = Algebraic_Function_Model_Bayesian_p_Value)
  
  
  # Generating a More General Algebraic Function Model
  
  # Response = ((0.5 * ((Intercept + (Slope * Predictor)) / ((1 + (abs(Intercept + (Slope * Predictor)) ^ Exponent)) ^ (1 / Exponent)))) + 0.5)
  
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
  A_More_General_Algebraic_Function_Model_Sum_of_Squared_Residuals <- sum((Response - ((0.5 * ((as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) / ((1 + (abs(as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept) + as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope) * Predictor) ^ as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent))) ^ (1 / as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent))))) + 0.5)) ^ 2)
  A_More_General_Algebraic_Function_Model <- paste0(Response_Name, " = ((0.5 * ((", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) / ((1 + (abs(", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Intercept), " + (", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Slope), " * ", Predictor_Name, ")) ^ ", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent), ")) ^ (1 / ", as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Exponent), ")))) + 0.5)")
  A_More_General_Algebraic_Function_Model_Bayesian_p_Value <- as.numeric(A_More_General_Algebraic_Function_Model_Output$BUGSoutput$mean$Bayesian_p_Value)
  A_More_General_Algebraic_Function_Model_Information <- list(Model_Name = A_More_General_Algebraic_Function_Model_Name, Lowercase_Model_Name = Lowercase_A_More_General_Algebraic_Function_Model_Name, Model = A_More_General_Algebraic_Function_Model, Sum_of_Squared_Residuals = A_More_General_Algebraic_Function_Model_Sum_of_Squared_Residuals, Fitted_Response_Values = Fitted_A_More_General_Algebraic_Function_Model_Response_Values, Output = A_More_General_Algebraic_Function_Model_Output, Bayesian_p_Value = A_More_General_Algebraic_Function_Model_Bayesian_p_Value)
  
  
  # Compiling the Models Into One List
  
  Model_List <- list(Logistic_Function_Model = Logistic_Function_Model_Information, Hyperbolic_Tangent_Model = Hyperbolic_Tangent_Model_Information, Arctangent_Function_Model = Arctangent_Function_Model_Information, Gudermannian_Function_Model = Gudermannian_Function_Model_Information, Error_Function_Model_Information = Error_Function_Model_Information, Generalised_Logistic_Function_Model = Generalised_Logistic_Function_Model_Information, Algebraic_Function_Model = Algebraic_Function_Model_Information, A_More_General_Algebraic_Function_Model = A_More_General_Algebraic_Function_Model_Information)
  
  
  # Returning the Pertinent Model Information
  
  Pertinent_Model_Information_List <- lapply(Model_List, function (x) {
    list(Model_Name = x$Model_Name, Model = x$Model, Fitted_Values = data.frame(Predictor = Fitted_Predictor_Values, Response = x$Fitted_Response_Values), Sum_of_Squared_Residuals = x$Sum_of_Squared_Residuals, Output = x$Output$BUGSoutput$summary)
  })
  Pertinent_Model_Information_List <- list(Model_Information = Pertinent_Model_Information_List, Conclusion = paste0("The model that best fits the data is the ", unlist(sapply(Model_List, `[`, 'Lowercase_Model_Name'))[which.min(unlist(sapply(Model_List, `[`, 'Sum_of_Squared_Residuals')))], " model."))
  return (Pertinent_Model_Information_List)
  
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

# Generating a Plot of All the Models

# Make sure to expand the plotting window.

Color <- 2:(length(Function_Output$Model_Information) + 1)
par(mar = c(12, 4, 4, 2))
plot(Response_Variable ~ Predictor_Variable, Data_Frame, main = "Fitting Sigmoid Models to the Data", xlab = "", ylab = "")
title(xlab = "Predictor Variable", line = 2.5)
title(ylab = "Response Variable", line = 2.5)
for (i in seq_len(length(Function_Output$Model_Information))) {
  lines(Function_Output$Model_Information[[i]]$Fitted_Values$Response ~ Function_Output$Model_Information[[i]]$Fitted_Values$Predictor, col = Color[i])
}
legend("bottom", xpd = T, ncol = 2, inset = c(0, -0.55), title = "Model Type (Sum of Squared Residuals)", legend = paste0(unlist(sapply(Function_Output$Model_Information, `[`, 'Model_Name')), " (", round(unlist(sapply(Function_Output$Model_Information, `[`, 'Sum_of_Squared_Residuals')), 3), ")"), col = Color, lty = 1)


# Works Cited

# Su, Y.-S., and M. Yajima. 2021. R2jags: Using R to Run 'JAGS'. R package
# version 0.7-1. https://cran.r-project.org/web/packages/R2jags/.
