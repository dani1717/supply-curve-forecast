# This script creates four lists: subir, subir_test, bajar, bajar_test
# Each one is composed of four data.tables: prices, offers, reqs (requirements), finalPrices
# the _test ones are from 2019 onwards while subir and bajar are older information for training

library(readxl)      # read_excel
library(parallel)    # mcmapply
library(data.table)  # setDT
library(dplyr)       # anti_join

rm(list = ls())

# Files ------------------------------------------------------------------------
  secundarioaSubir <- "~/Máster Big Data/TFM/Data/MercadoSecundarioSubir.RData"
  secundarioaSubir_reqs <- "~/Máster Big Data/TFM/Data/RequerimientosSecundariaASubir.xlsx"
  secundarioaBajar <- "~/Máster Big Data/TFM/Data/MercadoSecundarioBajar.RData"
  secundarioaBAjar_reqs <- "~/Máster Big Data/TFM/Data/RequerimientosSecundariaABajar.xlsx"

# Load SUBIR + its requirements ------------------------------------------------
  load(secundarioaSubir)
  offer.all[['date']] <- as.integer(offer.all[['date']])
  price.all[['date']] <- as.integer(price.all[['date']])
  weekDays <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
  price.all[['weekday']] <- factor(price.all[['weekday']], levels = weekDays, labels=weekDays)
  offer.all[['weekday']] <- factor(offer.all[['weekday']], levels = weekDays, labels=weekDays)
  
  # We read the requirements
  data_prov <- read_excel(secundarioaSubir_reqs)
  reqs <- list()
  reqs[['date']] <- unlist(lapply(data_prov[['datetime']], function(x) {
    date_prov <- substr(x, 1, 10)
    date_prov <- as.Date(date_prov)
    date_prov <- format(date_prov, "%Y%m%d")
    date_prov <- as.integer(date_prov)
    return(date_prov)
  }))
  reqs[['hour']] <- unlist(lapply(data_prov[['datetime']], function(x) {
    hour <- substr(x, 12,13)
    hour <- as.integer(hour) +1
    return(hour)
  }))
  reqs[['req']] <- data_prov[['value']]
  
  subir <- list(data.frame(price.all),data.frame(offer.all),data.frame(reqs))
  names(subir) <- c('prices','offers','reqs')
  subir[['reqs']] <- subir[['reqs']][complete.cases(subir[['reqs']]),]
  
  # Create indexes
  setDT(subir[[1]], key = c("date", "hour"))
  setDT(subir[[2]], key = c("date", "hour"))
  setDT(subir[[3]], key = c("date", "hour"))

# Load BAJAR + its requirements ------------------------------------------------
  load(secundarioaBajar)
  offer.all[['date']] <- as.integer(offer.all[['date']])
  price.all[['date']] <- as.integer(price.all[['date']])
  price.all[['weekday']] <- factor(price.all[['weekday']], levels = weekDays, labels=weekDays)
  offer.all[['weekday']] <- factor(offer.all[['weekday']], levels = weekDays, labels=weekDays)
  
  # We read the requirements
  data_prov <- read_excel(secundarioaBAjar_reqs)
  reqs <- list()
  reqs[['date']] <- unlist(lapply(data_prov[['datetime']], function(x) {
    date_prov <- substr(x, 1, 10)
    date_prov <- as.Date(date_prov)
    date_prov <- format(date_prov, "%Y%m%d")
    date_prov <- as.integer(date_prov)
    return(date_prov)
  }))
  reqs[['hour']] <- unlist(lapply(data_prov[['datetime']], function(x) {
    hour <- substr(x, 12,13)
    hour <- as.integer(hour) +1
    return(hour)
  }))
  reqs[['req']] <- data_prov[['value']]
  
  bajar <- list(data.frame(price.all),data.frame(offer.all),data.frame(reqs))
  names(bajar) <- c('prices','offers','reqs')
  bajar[['reqs']] <- bajar[['reqs']][complete.cases(bajar[['reqs']]),]
  
  # Create indexes
  setDT(bajar[[1]], key = c("date", "hour"))
  setDT(bajar[[2]], key = c("date", "hour"))
  setDT(bajar[[3]], key = c("date", "hour"))
  
  # Remove other objects in workspace
  rm(list = ls()[which(ls() %in% c('subir', 'bajar') == FALSE)])


# Imputation of missing data --------------------------------------------------

  # The requirements for 03-03-2016 are missing, so we impute them the avg of the surrounding days
  reqs_imput <- data.table(date=rep(20160303,24),hour=1:24,req=mapply(mean,subir[['reqs']][.(20160302,1:24),][[3]],subir[['reqs']][.(20160304,1:24),][[3]]))
  subir[['reqs']] <- rbind(subir[['reqs']],reqs_imput)
  setorder(subir[['reqs']], date, hour)
  reqs_imput <- data.table(date=rep(20160303,24),hour=1:24,req=mapply(mean,bajar[['reqs']][.(20160302,1:24),][[3]],bajar[['reqs']][.(20160304,1:24),][[3]]))
  bajar[['reqs']] <- rbind(bajar[['reqs']],reqs_imput)
  setorder(bajar[['reqs']], date, hour)
  rm(reqs_imput)
  setDT(subir[[3]], key = c("date", "hour"))
  setDT(bajar[[3]], key = c("date", "hour"))
  
  # We have some data missing due to hour change in spring, we assign the previous day value
  missing_rows <- anti_join(subir[['prices']], subir[['reqs']], by = c("date", "hour"))
  new_reqs <- data.table(subir[['reqs']][.(missing_rows$date-1,missing_rows$hour)])
  new_reqs[, date := date + 1]
  subir[['reqs']] <- rbind(subir[['reqs']],new_reqs)
  setorder(subir[['reqs']], date, hour)
  
  missing_rows <- anti_join(bajar[['prices']], bajar[['reqs']], by = c("date", "hour"))
  new_reqs <- data.table(bajar[['reqs']][.(missing_rows$date-1,missing_rows$hour)])
  new_reqs[, date := date + 1]
  bajar[['reqs']] <- rbind(bajar[['reqs']],new_reqs)
  setorder(bajar[['reqs']], date, hour)
  rm(missing_rows, new_reqs)
  
  setDT(subir[[3]], key = c("date", "hour"))
  setDT(bajar[[3]], key = c("date", "hour"))
  
  # We have duplicates for autumn hour change
  subir[['reqs']] <- dcast(subir[['reqs']], date + hour ~ ., fun.aggregate = mean, value.var = "req")
  bajar[['reqs']] <- dcast(bajar[['reqs']], date + hour ~ ., fun.aggregate = mean, value.var = "req")
  setDT(subir[[3]], key = c("date", "hour"))
  setDT(bajar[[3]], key = c("date", "hour"))
  
# Constants --------------------------------------------------------------------
  ncols <- length(subir[['prices']][2,]) 
  paleta <- paletteer::paletteer_d("ggthemes::Classic_Green_Orange_12")
  paleta <- c("dodgerblue2", "#E31A1C","green4", "#6A3D9A",  "#FF7F00", "gold1", "skyblue2", "#FB9A99",  "palegreen2", "#CAB2D6", "#FDBF6F", "gray70", "khaki2", "maroon", "orchid1", "deeppink1", "blue1", "steelblue4","darkturquoise", "green1", "yellow4", "yellow3", "darkorange4", "brown")
  paleta <- paleta[sample(1:25)]
  paleta <- rep(paleta,times=5)

# Add final prices -------------------------------------------------------------
  
  getFinalPrice <- function(data,date0,hour0) {
    # Returns the equilibrium price (intersection between the offer and the requirement)
  
    req <- data[['reqs']][.(date0,hour0),3][[1]]
    col <- findInterval(req,na.omit(unlist(unname(data[['offers']][.(date0,hour0),4:ncols])))) + 4
    #print(paste(date0,hour0))
    return(data[['prices']][.(date0,hour0),..col][[1]])
  }
  
  # Subir 
  finalPrices <- list()
  finalPrices[['date']] <- subir[['reqs']]$date
  finalPrices[['hour']] <- subir[['reqs']]$hour
  finalPrices[['finalPrice']] <- mapply(function(date,hour) getFinalPrice(subir,date,hour), finalPrices[['date']], finalPrices[['hour']])
  subir[['finalPrices']] <- data.frame(finalPrices)
  setDT(subir[[4]], key = c("date", "hour"))
  
  # Bajar
  finalPrices <- list()
  finalPrices[['date']] <- bajar[['reqs']]$date
  finalPrices[['hour']] <- bajar[['reqs']]$hour
  finalPrices[['finalPrice']] <- mapply(function(date,hour) getFinalPrice(bajar,date,hour), finalPrices[['date']], finalPrices[['hour']])
  bajar[['finalPrices']] <- data.frame(finalPrices)
  setDT(bajar[[4]], key = c("date", "hour"))

  # There are some times when the requirement is higher than the offers so there is no finalPrice for those rows. We impute them as the average of previous and following days (same hour)
  missing_rows <- which(apply(subir[['finalPrices']], 1, function(x) any(is.na(x))))
  write.csv(subir[['finalPrices']][missing_rows,1:2], file = 'Data/noMatchDates_subir.csv', row.names = FALSE)
  missing_rows_sum <- length(missing_rows)
  for (i in missing_rows) {
    subir[['finalPrices']][i,3] <- mean(c(subir[['finalPrices']][i-24,3][[1]],subir[['finalPrices']][i+24,3][[1]]))
  }
  missing_rows <- which(apply(bajar[['finalPrices']], 1, function(x) any(is.na(x))))
  missing_rows_sum <- missing_rows_sum + length(missing_rows)
  for (i in missing_rows) {
    bajar[['finalPrices']][i,3] <- mean(c(bajar[['finalPrices']][i-24,3][[1]],bajar[['finalPrices']][i+24,3][[1]]))
  }
  
  print(paste0(missing_rows_sum,' finalPrices were imputed since the requirement for those days was higher than the offers'))
  rm(finalPrices,getFinalPrice,missing_rows,missing_rows_sum,i)
  
# Save workspace ---------------------------------------------------------------
  
  save.image("Data/01_Load_data_workspace.RData")
  print('.RData was saved')
  
  write.csv(subir[['reqs']], file = 'Data/requirements_subir.csv', row.names = FALSE)
  write.csv(subir[['finalPrices']], file = 'Data/finalPrices_subir.csv', row.names = FALSE)
  
  
  
  
  
  
  
  
  
  
  
  
  