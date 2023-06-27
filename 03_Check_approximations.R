

# LOAD DATA -----------------------------------------------------------------------

  rm(list=ls())
  library(data.table)  # setDT
  set.seed(17)
  
  load("Data/02_Approximations.RData")
  ncols <- ncol(subir[['offers']])
  # data.table objects are stored in workspace without index. We must define it again
  setDT(subir[[1]], key = c("date", "hour"))
  setDT(subir[[2]], key = c("date", "hour"))
  setDT(subir[[3]], key = c("date", "hour"))
  setDT(subir[[4]], key = c("date", "hour"))
  setDT(bajar[[1]], key = c("date", "hour"))
  setDT(bajar[[2]], key = c("date", "hour"))
  setDT(bajar[[3]], key = c("date", "hour"))
  setDT(bajar[[4]], key = c("date", "hour"))

  approx_subir <- read.csv("Data/approxs_subir.csv")
  setDT(approx_subir, key = c("date", "hour"))
  approx_bajar <- read.csv("Data/approxs_bajar.csv")
  setDT(approx_bajar, key = c("date", "hour"))
  p <- read.csv("Data/p.csv")[['x']]
  n <- length(p)

# PLOT APPROXIMATIONS --------------------------------------------------------
  
  getApproxPrice <- function(approx_data,date0,hour0,req) {
    # Returns the equilibrium price between the curve of approx_data for that day and the requirement
    
    offers <- unname(approx_data[.(date0,hour0),4:n])
    i <- findInterval(req,offers) + 1
    return(p[i])
  }
  
  plotComparison <- function(data,approx_data,date0=-1,hour0=-1,xlimRight=-1) {
    # Plots the true curve and the approximation
    
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
    lines(getStepFun(p,approx_data[.(date0,hour0),4:(n+3)]),col='#F80303')
    # add requirement (horizontal line)
    abline(h = req, col = "black")
    # add finalPrice (Vertical segment)
    price_real = data[['finalPrices']][.(date0,hour0),3][[1]]
    segments(x0 = price_real, y0 = -4, x1 = price_real, y1 = req,col = "#E07700")
    # add price approximation (Vertical segment)
    price_approx <- getApproxPrice(approx_data,date0,hour0,req)
    segments(x0 = price_approx, y0 = -4, x1 = price_approx, y1 = req,col = "#FDEE00")
    # add legend and texts
    legend("bottomright", c('Original curve','Approx',"Req", "Price_real",'Price_approx'), lty = 1, col = c('#141F89','#F80303',"black", "#E07700",'#FDEE00'), cex=0.75, bty="n")
    mtext(paste("Final Price",round(price_real,2),'€/MWh'), side = 1, line = -1, cex = 0.8, col='#F55A00')
    mtext(paste("Error:",round(price_real-price_approx,2),'€'), side = 1, line = 0, cex = 0.8, col='#A25021')
    
  }
  
  plotComparison(subir,approx_subir)

# CHECK APPROXIMATIONS -------------------------------------------------------
  
  areaBetween <- function(data,approx_data,date0,hour0) {
    # Returns the area between both curves in the range p[1] until p[n]
    
    f_orig <- getStepFun(data[['prices']][.(date0,hour0),4:ncols],data[['offers']][.(date0,hour0),4:ncols])
    #plot(f_orig,xlim=c(0,500))
    f_approx <- getStepFun(p,approx_data[.(date0,hour0),4:(n+3)])
    #lines(f_approx,col='red')
    difference_abs <- function(x) {abs(f_orig(x)-f_approx(x))}
    #plot(difference_abs,xlim=c(0,500))
    return(c(date0,hour0,integrate(difference_abs,lower=p[1],upper=p[n],subdivisions = 50000,stop.on.error = FALSE)$value))
  }
  
  areaBetween(subir,approx_subir,20211225,12)

  dates <- subir[['prices']][,1][[1]]
  hours <- subir[['prices']][,2][[1]]
  diff_areas <- mapply(areaBetween, MoreArgs = list(data = subir, approx_data = approx_subir), date0 = dates, hour0 = hours, SIMPLIFY = FALSE)
  diff_areas <- data.frame(do.call(rbind, diff_areas))
  colnames(diff_areas) <- c('date','hour','area')
  diff_areas <- diff_areas[order(-diff_areas$area), ]
  diff_areas_total <- diff_areas
  for (i in 1:2) {
    plotComparison(subir,approx_subir,diff_areas[i,1],diff_areas[i,2])
  }  
  
  dates <- bajar[['prices']][,1][[1]]
  hours <- bajar[['prices']][,2][[1]]
  diff_areas <- mapply(areaBetween, MoreArgs = list(data = bajar, approx_data = approx_bajar), date0 = dates, hour0 = hours, SIMPLIFY = FALSE)
  diff_areas <- data.frame(do.call(rbind, diff_areas))
  colnames(diff_areas) <- c('date','hour','area')
  diff_areas <- diff_areas[order(-diff_areas$area), ]
  diff_areas_total <- rbind(diff_areas,diff_areas_total)
  for (i in 1:2) {
    plotComparison(bajar,approx_bajar,diff_areas[i,1],diff_areas[i,2])
  } 
  
  boxplot(diff_areas_total$area)
  plotComparison(subir,approx_subir,20211225,12,xlimRight = 1000)
  
  # diff areas total salen errores grandes y solo 5.800 obs, tendrían q salir N*2
  