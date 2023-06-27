

# LOAD DATA --------------------------------------------------------------------
  
  library(stringr)     # str_to_title
  library(data.table)  # setDT
  library(MASS)        # fitdistr
  
  rm(list=ls())
  set.seed(17)
  load("Data/01_Load_data_workspace.RData")
  # data.table objects are stored in workspace without index. We must define it again
  setDT(subir[[1]], key = c("date", "hour"))
  setDT(subir[[2]], key = c("date", "hour"))
  setDT(subir[[3]], key = c("date", "hour"))
  setDT(subir[[4]], key = c("date", "hour"))
  setDT(bajar[[1]], key = c("date", "hour"))
  setDT(bajar[[2]], key = c("date", "hour"))
  setDT(bajar[[3]], key = c("date", "hour"))
  setDT(bajar[[4]], key = c("date", "hour"))

# PLOTTING FUNCTIONS  ----------------------------------------------------------
  
  getStepFun <- function(x,y,max_prices=9999) {
    # Returns the step function extrapolated from the points x (prices), y(offers)
    
    # We eliminate NA's and add 0 value to y
    x <- unname(unlist(x))
    x <- c(x[!is.na(x)],max_prices*2)
    y <- unname(unlist(y))
    y <- c(0,unlist(y[!is.na(y)]))
    y <- c(y, tail(y, 1))
    return(stepfun(x,y))
  }
  
  plotCurves <- function(lista, dates, hours=1:24) {
    # Plots in a graph the curves from subir or bajar (lista) given some dates and hours
    heading <- paste(deparse(substitute(lista)),paste(dates, collapse=" "))
    prices <- lista[['prices']][.(dates,hours),4:ncols]
    offers <- lista[['offers']][.(dates,hours),4:ncols]
    max_prices <- max(sapply(prices, max, na.rm = TRUE))
    max_offers <- max(sapply(offers, max, na.rm = TRUE))
    plot(NA, xlim = c(0,max_prices), ylim = c(0,max_offers), xlab='€/MWh', ylab='MWh', main=heading)
    for (i in 1:dim(prices)[[1]]) {
      lines(getStepFun(prices[i,],offers[i,]), col=paleta[i], lwd=1.75)
    }
  }

# GENERATE p  -------------------------------------------------
  
  n <- 50
  
  # q0 for the conditional distribution
  # all_offers <- c(na.omit(unlist(subir[['offers']][,4:ncols],use.names = FALSE)),na.omit(unlist(bajar[['offers']][,4:ncols],use.names = FALSE)))
  # q0 = quantile(all_offers,0.25)
  # rm(all_offers)
  q0_default = 337  # first quartile
  
  generate_p <- function(n, cdf_prices=cdf_prices_conditional()) {
    # Based on some cumulative distribution function of prices (f) creates a sample of n prices
    
    cdf_inv <- function(x) unname(quantile(cdf_prices,x))   # inverse of the cdf
    p <- list()
    extra_points <- 1
    while (length(p)<n) {
      # Sometimes we obtain repeated values or values very high in p so we loop until we find at least n different prices
      p <- unique(cdf_inv(seq(from=0,to=1,length.out=n+extra_points)))
      p <- p[p!=0]
      p <- p[p<1000]
      extra_points <- extra_points +1
    }
    p <- p[1:n]
    
    # Plot results
    dens <- density(p, from = 0, to = 700)
    y_max <- max(dens$y)
    plot(NA, col = "white", xlim=c(0,500),ylim=c(0,y_max), ylab='density', xlab='price', main=paste('Density of generated prices \nn=',length(p)))
    invisible(lapply(p, function(x) abline(v=x, col="#EB7676")))
    lines(dens, col = "#BE0F0F", lwd=3)
    
    return(p)
  }
  
  cdf_prices_conditional <- function(q0=q0_default) {
    # Returns the cdf of all prices that meet the condition of its q>=q0
    
    indices <- unlist(which(bajar[['offers']][,4:ncols] >= q0, arr.ind = TRUE))
    df_prices <- data.frame(bajar[["prices"]][,4:ncols])
    prices_cond <- df_prices[indices]
    indices <- unlist(which(subir[['offers']][,4:ncols] >= q0, arr.ind = TRUE))
    df_prices <- data.frame(subir[["prices"]][,4:ncols])
    prices_cond <- c(prices_cond,df_prices[indices])
    return(ecdf(prices_cond))
  }
  
  cond_prices_unique <- function(){
    p.uniques <- list()
    p.uniques <- na.omit(unlist(apply(subir[['prices']][,4:ncols], 1, unique)))
    p.uniques <- c(p.uniques,na.omit(unlist(apply(bajar[['prices']][,4:ncols], 1, unique))))
    return(ecdf(p.uniques))
  }

  cond_prices_unique_q0 <- function(q0=q0_default){
    data <- subir
    p.uniques <- list()
    for (i in 1:nrow(data[['prices']])) {
      Q = approxfun(data[['prices']][i,4:ncols], data[['offers']][i,4:ncols], method = "constant", rule = 2, ties = max, na.rm = T)
      q.uniques <- Q(na.omit(unique(unlist(data[['prices']][i,4:ncols]))))
      q.uniques <- q.uniques[q.uniques >= q0]
      indices <-which(!is.na(match(data[['offers']][i,4:ncols],q.uniques)))
      p.uniques <- c(p.uniques, unlist(data[['prices']][i,4:ncols])[indices])
      
    }
    # data <- bajar
    # for (i in 1:nrow(data[['prices']])) {
    #   Q = approxfun(data[['prices']][i,4:ncols], data[['offers']][i,4:ncols], method = "constant", rule = 2, ties = max, na.rm = T)
    #   q.uniques <- c(q.uniques,Q(na.omit(unique(unlist(data[['prices']][i,4:ncols])))))
    # }
    # q.uniques <- unique
    # return(q.uniques)
    return(q.uniques)
  }

  cond_prices_unique_q02 <- function(q0=q0_default){
    data <- subir
    
    # function to calculate the right q's for a row of prices (we will use it later in apply)
    calcular_q_uniques <- function(prices) {
      Q <- approxfun(prices, data[['offers']][i,4:ncols], method = "constant", rule = 2, ties = max, na.rm = TRUE)
      q_uniques <- Q(unique(prices[!is.na(prices)]))
      q_uniques[q_uniques >= q0]
    }
    
    # Calcular los valores únicos de q para cada fila
    q_uniques_list <- lapply(data[['prices']][,4:ncols], calcular_q_uniques)
    # DA ERROR CREO QUE ES POR LA i QUE USA APPROXFUN, ESTE APPLY ESTÁ MAL
    
    # Encontrar los índices de los elementos de q.uniques en data[['offers']][i,4:ncols]
    indices_list <- mapply(function(q_uniques, offers) which(offers %in% q_uniques), q_uniques_list, data[['offers']][,4:ncols], SIMPLIFY = FALSE)
    
    # Obtener los valores correspondientes de prices para cada conjunto de índices
    p_uniques_list <- mapply(function(prices, indices) prices[indices], data[['prices']][,4:ncols], indices_list, SIMPLIFY = FALSE)
    
    # Combinar todos los valores de p.uniques en un solo vector
    p_uniques <- unlist(p_uniques_list)
    
    return(p_uniques)
  }
  
  p <- generate_p(n)

# APROXIMATIONS -----------------------------------------
  
  # W_uniquePrices_exp
  exp_rate <- 0.02407047
  W <- function(x) dexp(x, rate = exp_rate)
  
  CW <- function(C,W,i,lastTerm=FALSE) {
    ifelse(lastTerm,upper<-Inf,upper<-p[i+1])
    integrate(function(x) C(x)*W(x), lower = p[i], upper = upper, subdivisions = 10000, stop.on.error = FALSE)$value
  }
  W_i <- function(W,i,lastTerm=FALSE) {
    ifelse(lastTerm,upper<-Inf,upper<-p[i+1])
    integrate(W, lower = p[i], upper = upper, subdivisions = 10000)$value
  }
  
  compute_c_star <- function(x,y) {
    # Returns the y-approximations (c*) of the curve defined by the prices x and offers y
    # c* corresponds to p
    
    x <- na.omit(unlist(unname(as.list(x))))
    y <- na.omit(unlist(unname(as.list(y))))
    C <- getStepFun(x,y)
    c_star <- c()
    n <- length(p)
    
    for (i in 1:(n-1)) {
      denom <- W_i(W,i)
      if (denom != 0) {
        c_new <- round(CW(C,W,i)/denom,5)
      }
      c_star <- c(c_star,c_new)
    }
    denom <- W_i(W,n,lastTerm=TRUE)
    if (denom != 0) {
      c_new <- CW(C,W,n,lastTerm=TRUE)/denom
    }
    c_star <- c(c_star,c_new)
    return(c_star)
  }
  
  c_star_byDate <- function(data,date0,hour0) {
    # Returns the approximation function for that data, date and hour
    return(compute_c_star(data[['prices']][.(date0,hour0),4:ncols],data[['offers']][.(date0,hour0),4:ncols]))
  }

  getApproxsByDatesHours <- function(data, datesHours) {
    # Returns a dataframe with the approximation functions of data (subir or bajar) for those datesHours
    # datesHours is a dataframe with two columns (date, hour)
    
    results <- data.frame(matrix(nrow = dim(datesHours)[1], ncol = n))
    results <- cbind(date=datesHours$date,hour=datesHours$hour,results)
    results[, 3:(n+2)] <- do.call(rbind,mapply(c_star_byDate, data = list(data), date = datesHours$date, hour = datesHours$hour, SIMPLIFY = FALSE))
    return(results)
  }
  

# ERROR  ---------------------------------------------------------
  
  getPrice_approx_aux <- function(data, row) {
    # Returns the equilibrium price between the curve in row and the requirements in data
    # row is composed of date, hour, and values of the function for every p[i]
    
    date0 <- row[[1]]
    hour0 <- row[[2]]
    req <- data[['reqs']][.(date0,hour0),3][[1]]
    i <- findInterval(req,na.omit(unname(row[3:n]))) + 1
    return(p[i])
  }
  
  getPrice_approx <- function(data, df, onlyPrices=FALSE) {
    # Returns a list with the equilibrium prices between the curve in each row of df and the requirements in data (subir or bajar)
    # df is a dataframe composed of date,hour and n columns with approximations of curves
    prices <- unlist(lapply(seq_len(nrow(df)), function(i) getPrice_approx_aux(data, df[i,])))
    if (onlyPrices) {
      return(prices)
    }
    else {
      results = data.frame(date=df[1],hour=df[2],prices_approx=prices)
      return(results)
    }
  }
  
  getAvgError <- function(data, df) {
    # Returns the avg error in the prices of the approximations in the dataframe df
    # df is a dataframe of approx functions, like the result of getPrice_approx
    
    data_sub <- data[['finalPrices']][.(df$date, df$hour)]
    error_sq <- (df[,3] - data_sub$finalPrice)^2
    return(sqrt(sum(error_sq)/nrow(df)))
  }
  
  check_approx <- function(data,date0=-1,hour0=-1,xlimRight=-1) {
    # Random date and hour if not given
    if (date0 == -1) {
      date0 <- sample(data[['finalPrices']]$date,1)
      hour0 <- sample(1:24,1)
    }
    # getting prices and offers for that date
    prices <- data[['prices']][.(date0,hour0),4:ncols]
    offers <- data[['offers']][.(date0,hour0),4:ncols]
    # plotting parameters
    max_prices <- ifelse(xlimRight==-1, max(prices,na.rm = TRUE), xlimRight)
    req <- data[['reqs']][.(date0,hour0),3][[1]]
    max_offers <- max(req,max(offers,na.rm = TRUE))
    heading <- paste('Approx for',date0,hour0)
    #plot original function
    plot(getStepFun(prices,offers), xlim = c(0,max_prices), ylim = c(0,max_offers), col='#141F89', lwd=2, xlab='€/MWh', ylab='MWh', main=heading)
    # add approximation
    lines(getStepFun(p,c_star_byDate(data,date0,hour0)),col='#F80303')
    # add requirement (horizontal line)
    abline(h = req, col = "black")
    # add finalPrice (Vertical segment)
    price_real = data[['finalPrices']][.(date0,hour0),3][[1]]
    segments(x0 = price_real, y0 = -4, x1 = price_real, y1 = req,col = "#E07700")
    # add price approximation (Vertical segment)
    approximationFun_row <- c(date0,hour0,c_star_byDate(data,date0 = date0,hour0=hour0))
    price_approx <- getPrice_approx_aux(data,approximationFun_row)
    segments(x0 = price_approx, y0 = -4, x1 = price_approx, y1 = req,col = "#FDEE00")
    # add legend and texts
    legend("bottomright", c('Original curve','Approx',"Req", "Price_real",'Price_approx'), lty = 1, col = c('#141F89','#F80303',"black", "#E07700",'#FDEE00'), cex=0.75)
    mtext(paste("Final Price",round(price_real,2),'€/MWh'), side = 1, line = -1, cex = 0.8, col='#F55A00')
    mtext(paste("Error:",round(price_real-price_approx,2),'€'), side = 1, line = 0, cex = 0.8, col='#A25021')
  }
  
  check_approx(subir)

# PRUEBAS ---------------------------------------------------------
  
  sampleError <- function(data, sampleSize=200) {
    # Takes a sample of sampleSize curves, computes the approximations and measures the 
    # avg error of the approximated price to the real final price. data is subir or bajar
    
    # We sample some dates and hours and join them in a dataframe
    dates0 <- sample(data[['finalPrices']]$date,sampleSize)
    hours0 <- sample(1:24,sampleSize,replace = TRUE)
    datesHours <- data.frame(date=dates0,hour=hours0)
    # We get a dataframe with our approximation curves for those dates and hours
    aprox_curves <- getApproxsByDatesHours(data,datesHours)
    # We get the prices using those approximations curves
    aprox_prices <- getPrice_approx(data,aprox_curves)
    # We get the real prices
    real_prices <- data[['finalPrices']][.(dates0, hours0)]
    # We compute the square error
    error_sq <- (aprox_prices[,3] - real_prices$finalPrice)^2
    # We return the avg error
    return(sqrt(sum(error_sq,na.rm = TRUE)/length(error_sq)))
  }
  
  nSamplesError <- function(data,nSamples=5, sampleSize=200) {
    # Computes several times the avg error for a sample and saves some statistics in a file
    errors <- c()
    for (i in 1:nSamples) {
      errors <- c(errors,sampleError(data,sampleSize))
    }
    errors <- na.omit(errors)
    results <- c(round(mean(errors,na.rm=TRUE),2),round(median(errors,na.rm=TRUE),2),round(sd(errors,na.rm=TRUE),2),quantile(errors,probs=c(0.25,0.75,0.9,0.95,0.99)))
    names(results) <- c('avgError','median','sd','Q1','Q3','P90','P95','P99')
    print(W)
    return(results)
  }

  
  # n
  nSamplesError_n <- function(nSamples=5, sampleSize=200,range_n){
    # for different values of n
    
    results <- data.frame()
    errors <- vector("list", length(range_n))
    for (i in seq_along(range_n)) {
      n <<- range_n[i]
      p <<- generate_p(n)
      for (j in 1:nSamples) {
        errors[[i]] <- c(errors[[i]],sampleError(subir,sampleSize))
      }
      results <- rbind(results,c(n,mean(errors[[i]],na.rm=TRUE),median(errors[[i]],na.rm=TRUE),sd(errors[[i]],na.rm=TRUE),quantile(errors[[i]],probs=c(0.25,0.75,0.9,0.95,0.99))))
      print(c(n,mean(errors[[i]],na.rm=TRUE)))
    }
    names <-  as.character(paste0("n=", range_n[seq_along(range_n)]))
    boxplot(errors, xaxt = "n", main = "Errors for different values of n",col = paleta)
    axis(side = 1, at = 1:length(range_n), labels = names)
    colnames(results) <- c('n','avgError','median','sd','Q1','Q3','P90','P95','P99')
    return(results)
  }
  
  range_n <- seq(from = 35, to = 55, by = 5)
  results_n <- nSamplesError_n(nSamples = 10000,1,range_n)
  par(mar=c(5.1, 4.1, 4.1, 2.1))
  plot(results_n$n, results_n$avgError, xlab = "n", ylab = "avgError(€)",type = 'l', main ='Error vs. n')
  points(results_n$n, results_n$avgError, pch = 19, col = "blue",cex=0.75)
  save(results_n, file = "results_n_dataframe2.Rdata")

  
  # q0
  
  nSamplesError_q0 <- function(nSamples=5, sampleSize=200,range_q0){
    # for different values of n
    
    results <- data.frame()
    errors <- vector("list", length(range_q0))
    for (i in seq_along(range_q0)) {
      p <<- generate_p(n, cdf_prices=cdf_prices_conditional(range_q0[i]))
      for (j in 1:nSamples) {
        errors[[i]] <- c(errors[[i]],sampleError(subir,sampleSize))
      }
      results <- rbind(results,c(range_q0[i],round(mean(errors[[i]],na.rm=TRUE),2),round(median(errors[[i]],na.rm=TRUE),2),round(sd(errors[[i]],na.rm=TRUE),2),quantile(errors[[i]],probs=c(0.25,0.75,0.9,0.95,0.99))))
      print(c(range_q0[i],mean(errors[[i]],na.rm=TRUE)))
    }
    names <-  as.character(paste0("q0=", range_q0[seq_along(range_q0)]))
    boxplot(errors, xaxt = "n", main = "Errors for different values of q0",col = paleta)
    axis(side = 1, at = 1:length(range_q0), labels = names)
    colnames(results) <- c('q0','avgError','median','sd','Q1','Q3','P90','P95','P99')
    return(results)
  }
  
  range_q0 <- c(0,150,337)
  results_q0 <- nSamplesError_q0(nSamples = 1000,1,range_q0)
  plot(results_q0$q0, results_q0$avgError, xlab = "q0", ylab = "avgError(€)",type = 'l', main ='Error vs. q0')
  points(results_q0$q0, results_q0$avgError, pch = 19, col = "blue",cex=0.75)
  save(results_q0, file = "Data/results_q0_dataframe.Rdata")
  load("Data/results_q0_dataframe.Rdata")
  
  
  # W
  W_functions = list()
  
  # First of all we study if final prices distr. are equal for subir and bajar so we can treat them together
  plot(density(subir[['finalPrices']][,3][[1]],na.rm = TRUE), col = "red", xlim = c(0,60), ylim = c(0,0.07), main = "Final prices",lwd=2)
  lines(density(bajar[['finalPrices']][,3][[1]],na.rm = TRUE), col = "blue",lwd=2)
  legend("topright", legend = c("subir", "bajar"), col = c("red", "blue"), lty = 1)
  
  par(mfrow=c(4,2))
  par(mar=c(1,1,1,1))
  # W = normal density function of final prices
  name_function <- 'W_finalPrices_Normal'
  finalPrices_all <- na.omit(c(subir[['finalPrices']][,3][[1]],bajar[['finalPrices']][,3][[1]]))
  fit1 <- fitdistr(finalPrices_all, 'normal')
  W <- function(x) dnorm(x,mean=fit1$estimate[[1]],sd = fit1$estimate[[2]])
  W_functions <- c(W_functions,W)
  names(W_functions) <- c(name_function)
  hist(finalPrices_all,xlim = c(0,85),breaks = 80,probability = TRUE, main=name_function)
  lines(W(1:1000))
  
  # W = exponential of the previous one
  name_function <- 'W_finalPrices_Normal_Exp'
  W <- function(x) exp(dnorm(x,mean=fit1$estimate[[1]],sd = fit1$estimate[[2]]))-1
  W_functions <- c(W_functions,W)
  names(W_functions)[length(W_functions)] <- name_function
  hist(finalPrices_all,xlim = c(0,85),breaks = 80,probability = TRUE, main=name_function)
  lines(W(1:1000))
  
  # W = Cauchy fit of finalPrices
  name_function <- 'W_finalPrices_cauchy'
  fit2 <- fitdistr(finalPrices_all, "cauchy")
  W <- function(x) dcauchy(x,location = fit2$estimate['location'],scale = fit2$estimate['scale'])
  W_functions <- c(W_functions,W)
  names(W_functions)[length(W_functions)] <- name_function
  hist(finalPrices_all,xlim = c(0,85),breaks = 80,probability = TRUE, main=name_function)
  lines(W(1:1000))
  
  # W = logNormal of positive finalPrices
  name_function <- 'W_finalPrices_logNormal'
  fit3 <- fitdistr(finalPrices_all[finalPrices_all>0], "lognormal")
  W <- function(x) dlnorm(x,meanlog = fit3$estimate['meanlog'],sdlog = fit3$estimate['sdlog'])
  W_functions <- c(W_functions,W)
  names(W_functions)[length(W_functions)] <- name_function
  hist(finalPrices_all,xlim = c(0,85),breaks = 80,probability = TRUE, main=name_function)
  lines(W(1:1000))
  
  # W = exponential adjusted to final prices
  name_function <- 'W_finalPrices_exp'
  fit4 <- fitdistr(finalPrices_all, "exponential")
  W <- function(x) dexp(x,rate = fit4$estimate['rate'])
  W_functions <- c(W_functions,W)
  names(W_functions)[length(W_functions)] <- name_function
  hist(finalPrices_all,xlim = c(0,85),breaks = 80,probability = TRUE, main=name_function)
  lines(W(1:1000))
  
  # W = exponential adjusted to unique prices
  name_function <- 'W_uniquePrices_exp'
  p.uniques <- list()
  p.uniques <- na.omit(unlist(apply(subir[['prices']][,4:ncols], 1, unique)))
  p.uniques <- c(p.uniques,na.omit(unlist(apply(bajar[['prices']][,4:ncols], 1, unique))))
  fit5 <- fitdistr(p.uniques, "exponential")
  W <- function(x) dexp(x,rate = fit5$estimate['rate'])
  W_functions <- c(W_functions,W)
  names(W_functions)[length(W_functions)] <- name_function
  hist(p.uniques,xlim = c(0,400),breaks = 1000,probability = TRUE, main=name_function)
  lines(W(1:1000))
  
  # W = exponential adjusted to all prices
  name_function <- 'W_allPrices_exp'
  p.all <- list()
  p.all <- na.omit(unlist(subir[['prices']][,4:ncols]))
  p.all <- c(p.all,na.omit(unlist(bajar[['prices']][,4:ncols])))
  fit6 <- fitdistr(p.all, "exponential")
  W <- function(x) dexp(x,rate = fit6$estimate['rate'])
  W_functions <- c(W_functions,W)
  names(W_functions)[length(W_functions)] <- name_function
  hist(p.all,xlim = c(0,400),breaks = 1000,probability = TRUE, main=name_function)
  lines(W(1:1000))
  par(mfrow=c(1,1))
  
  results_W <- data.frame()
  n <- 45
  q0 <- 150
  p <- generate_p(n)
  nSamples <- 10000
  for (i in 1:length(W_functions)) {
    W <- W_functions[[i]]
    results_W <- rbind(results_W,c(names(W_functions)[i],nSamplesError(subir,nSamples,1)))
  }
  results_W[,2:9] <- lapply(results_W[,2:9], as.numeric)
  results_W[,2:9] <- round(results_W[,2:9], 2)
  colnames(results_W) <- c('W','avgError','median','sd','Q1','Q3','P90','P95','P99')
  
  W <- W_functions[['W_uniquePrices_exp']]
  
# NAIVE ESTIMATORS ----------------------------------------------------------
  
  # We study the error in the finalPrice if we use two different naive approximators
  
  # 1st - Approximating curves with previous day curve (offers = offers - 24)
  estimateFinalPrice_withPreviousDay <- function(data,row) {
    # Computes the intersection of the requirement of that row with the curve of the previous day, same hour
    
    req <- data[['reqs']][row,3][[1]]
    if(row<=24) {return(NA)}
    col <- findInterval(req,na.omit(unlist(unname(data[['offers']][row-24,4:ncols])))) + 4
    return(data[['prices']][row-24,..col][[1]])
  }
  
  predictions_withPreviousDay <- vector(length = nrow(subir[['offers']]))
  for (i in 1:nrow(subir[['offers']])) {
    predictions_withPreviousDay[i] <- estimateFinalPrice_withPreviousDay(subir, i)
  }
  errors_withPreviousDay <- abs(subir[['finalPrices']][,3][[1]] - predictions_withPreviousDay)
  boxplot(errors_withPreviousDay, main='Errors approximating with previous day offers')
  errors_withPreviousDay_table <- c(round(mean(errors_withPreviousDay,na.rm=TRUE),2),round(median(errors_withPreviousDay,na.rm=TRUE),2),round(sd(errors_withPreviousDay,na.rm=TRUE),2),quantile(errors_withPreviousDay,probs=c(0.25,0.75,0.9,0.95,0.99),na.rm = TRUE))
  names(errors_withPreviousDay_table) <- c('avgError','median','sd','Q1','Q3','P90','P95','P99')
  errors_withPreviousDay_table
  
  # 2nd - Approximating price with previous day price (finalPrice = finalPrice - 24)
  predictions_lastPrice <- vector(length = nrow(subir[['offers']]))
  for (i in 1:nrow(subir[['offers']])) {
    predictions_lastPrice[i] <- ifelse(i>24,subir[['finalPrices']][i-24,3][[1]],NA)
  }
  errors_lastPrice <- abs(subir[['finalPrices']][,3][[1]] - predictions_lastPrice)
  boxplot(errors_lastPrice, main='Errors predicting previous day price')
  errors_lastPrice_table <- c(round(mean(errors_lastPrice,na.rm=TRUE),2),round(median(errors_lastPrice,na.rm=TRUE),2),round(sd(errors_lastPrice,na.rm=TRUE),2),quantile(errors_lastPrice,probs=c(0.25,0.75,0.9,0.95,0.99),na.rm = TRUE))
  names(errors_lastPrice_table) <- c('avgError','median','sd','Q1','Q3','P90','P95','P99')
  errors_lastPrice_table
  
# COMPUTE ALL APROXIMATIONS ------------------------------------------------------
  
  # W = exponential adjusted to unique prices
  name_function <- 'W_uniquePrices_exp'
  p.uniques <- list()
  p.uniques <- na.omit(unlist(apply(subir[['prices']][,4:ncols], 1, unique)))
  p.uniques <- c(p.uniques,na.omit(unlist(apply(bajar[['prices']][,4:ncols], 1, unique))))
  fit5 <- fitdistr(p.uniques, "exponential")
  W <- function(x) dexp(x,rate = fit5$estimate['rate'])
  
  n <- 50
  q0_default = 337
  p <- generate_p(n)
  
  exportApproximations <- function() {
    # Saves in Data/ the files p.csv, approxs_subir.csv and approxs_bajar.csv with the grid of prices and the approximations of the curves in that grid
    
    folder <- 'Data'
    # export p
    write.csv(p, file.path(folder,"p.csv"))
    t1 <- Sys.time()
    # export subir approximation curves
    results <- mapply(compute_c_star, as.data.frame(t(subir[['prices']][,4:ncols])), as.data.frame(t(subir[['offers']][,4:ncols])), SIMPLIFY = FALSE)
    results <- do.call(rbind, results)
    colnames(results) <- paste0("Q", 1:ncol(results))
    results <- cbind(subir[['prices']][,1:3],results)
    write.csv(results, file.path(folder,"approxs_subir.csv"), row.names = FALSE)
    print('Approximations for subir saved')
    # export bajar approximation curves
    results <- mapply(compute_c_star, as.data.frame(t(bajar[['prices']][,4:ncols])), as.data.frame(t(bajar[['offers']][,4:ncols])), SIMPLIFY = FALSE)
    results <- do.call(rbind, results)
    colnames(results) <- paste0("Q", 1:ncol(results))
    results <- cbind(bajar[['prices']][,1:3],results)
    write.csv(results, file.path(folder,"approxs_bajar.csv"), row.names = FALSE)
    print('Approximations for bajar saved')
    t2 <- Sys.time()
    print(t2-t1)
  }
  
  exportApproximations()

  save(bajar,subir,paleta,getStepFun,plotCurves, file = "Data/02_Approximations.Rdata")
  

# OTROS ---------------------------------------------------

# Histogram of all finalPrices
finalPrices_all <- c(subir[['finalPrices']][,3][[1]],bajar[['finalPrices']][,3][[1]])
hist(finalPrices_all,breaks = 80)
hist(finalPrices_all,xlim = c(0,85),breaks = 80)
boxplot(finalPrices_all)
summary(finalPrices_all)
knitr::kable(table(cut(finalPrices_all, seq(0, 225, by = 5))))
