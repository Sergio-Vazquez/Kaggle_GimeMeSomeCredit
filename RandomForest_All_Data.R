## PRACTICA APRENDIZAJE SUPERVISADO ##
############ OPCION 2  PARTE 1 - PRUEBA VARIOS MODELOS CON TODOS LOS DATOS ################

# Primero, descargamos los datos de kaggle: https://www.kaggle.com/c/GiveMeSomeCredit

# El objetivo es predecir si el cliente tendra problemas financieros en los proximos 2 años.
# La variable "SeriousDlqin2yrs" sera la variable a predecir



######### MODELO RANDOM FOREST CON EL 100% DE DATOS DE TRAIN - 500 TREES ############

# Cargamos las librerias:
library(randomForest)
library(caret)

# Creamos los conjuntos train y test
train.data <- read.csv("src/cs-training.csv")
test.data <- read.csv("src/cs-test.csv")
submissision <- read.csv("src/sampleEntry.csv")

# Pasamos a factor la variable a predecir:
train.data$SeriousDlqin2yrs <- as.factor(train.data$SeriousDlqin2yrs)

# Comprobamos resultados de la variable a predecir:
table(train.data$SeriousDlqin2yrs)
prop.table(table(train.data$SeriousDlqin2yrs)) # porcentaje de posibles impagos

# Creamos la clase predictora:
train.class <- train.data[["SeriousDlqin2yrs"]]

# Limpiamos los datos con valor NA (he probado con valores NA = 0 y NA = -10)
train.data[is.na(train.data[,12]), 12] <- -10  # aqui le doy -10 a las variables NA de "NumberOfDependences"
train.data[is.na(train.data[,7]), 7] <- -10 # aqui le doy -10 a las variables NA de "MonthlyIncome"

test.data[is.na(test.data[,12]), 12] <- -10  # aqui le doy -10 a las variables NA de "NumberOfDependences"
test.data[is.na(test.data[,7]), 7] <- -10 # aqui le doy -10 a las variables NA de "MonthlyIncome"


#### CREACION DE NUEVAS FEATURES ######

# PrestamosIngresos = NumberOfOpenCreditLinesAndLoans/MonthlyIncome
train.data$PrestamosIngresos <- (train.data$NumberOfOpenCreditLinesAndLoans/train.data$MonthlyIncome)
test.data$PrestamosIngresos <- (test.data$NumberOfOpenCreditLinesAndLoans/test.data$MonthlyIncome)
# Elimino valores NA y asigno valor a inf (he probado con valores inf = 50 e inf = 100)
train.data[is.na(train.data[,13]), 13] <- -10
train.data[is.infinite(train.data[,13]), 13] <- 50

test.data[is.na(test.data[,13]), 13] <- -10
test.data[is.infinite(test.data[,13]), 13] <- 50

# HipotecasIngresos = NumberRealEstateLoansOrLines/MonthlyIncome
train.data$HipotecasIngresos <- (train.data$NumberRealEstateLoansOrLines/train.data$MonthlyIncome)
test.data$HipotecasIngresos <- (test.data$NumberRealEstateLoansOrLines/test.data$MonthlyIncome)
# Elimino valores NA y asigno valor a inf
train.data[is.na(train.data[,14]), 14] <- -10
train.data[is.infinite(train.data[,14]), 14] <- 50

test.data[is.na(test.data[,14]), 14] <- -10
test.data[is.infinite(test.data[,14]), 14] <- 50

# Gastos = DebRatio * Montlhy Income
train.data$Gastos <- (train.data$DebtRatio * train.data$MonthlyIncome)
test.data$Gastos <- (test.data$DebtRatio * test.data$MonthlyIncome)

# Prestamos Total = NumberOfTime30-59DaysPastDueNotWorse + NumberOfTime60-89DaysPastDueNotWorse + NumberOfTimes90DaysLate

train.data$PrestamosTotal <- (train.data$NumberOfTime30.59DaysPastDueNotWorse + train.data$NumberOfTime60.89DaysPastDueNotWorse +
                                train.data$NumberOfTimes90DaysLate)
test.data$PrestamosTotal <- (test.data$NumberOfTime30.59DaysPastDueNotWorse + test.data$NumberOfTime60.89DaysPastDueNotWorse +
                               test.data$NumberOfTimes90DaysLate)



# MODELO: RF - TEST0 - 10 variables - original - NA = 0
rf.model.test0.part1 <- randomForest(train.data[,3:12], y = train.class, method = "class",ntree = 100) # class es el metodo de random forest, 
rf.model.test0.part2 <- randomForest(train.data[,3:12], y = train.class, method = "class",ntree = 100) # que utiliza el metodo de clasificacion.
rf.model.test0.part3 <- randomForest(train.data[,3:12], y = train.class, method = "class",ntree = 100)
rf.model.test0.part4 <- randomForest(train.data[,3:12], y = train.class, method = "class",ntree = 100)
rf.model.test0.part5 <- randomForest(train.data[,3:12], y = train.class, method = "class",ntree = 100)
rf.model.test0.all <- combine(rf.model.test0.part1,rf.model.test0.part2,rf.model.test0.part3,rf.model.test0.part4,rf.model.test0.part5)

pred.rf <- predict(rf.model.test0.all, test.data[,3:12], type = "prob") # predecimos en el modelo de test dado por kaggle, el modelo anterior

write.table(pred.rf[,2], file = "RF_TEST0_original.csv", quote = F, sep = ",") # con quote=F, no pone entre comillas el resultado



# MODELO: RF - TEST1 - 12 variables (2 features nuevas) con mtry - NA = 0
rf.model.test3.part1 <- randomForest(train.data[, -c(1,2,13,14)], y = train.class, method = "class",ntree = 100, mtry =  sqrt(12))
rf.model.test3.part2 <- randomForest(train.data[, -c(1,2,13,14)], y = train.class, method = "class",ntree = 100, mtry =  sqrt(12))
rf.model.test3.part3 <- randomForest(train.data[, -c(1,2,13,14)], y = train.class, method = "class",ntree = 100, mtry =  sqrt(12))
rf.model.test3.part4 <- randomForest(train.data[, -c(1,2,13,14)], y = train.class, method = "class",ntree = 100, mtry =  sqrt(12))
rf.model.test3.part5 <- randomForest(train.data[, -c(1,2,13,14)], y = train.class, method = "class",ntree = 100, mtry =  sqrt(12))
rf.model.test3.all <- combine(rf.model.test3.part1,rf.model.test3.part2,rf.model.test3.part3,rf.model.test3.part4,rf.model.test3.part5)

pred.rf <- predict(rf.model.test3.all, test.data[, -c(1,2,13,14)], type = "prob")

write.table(pred.rf[,2], file = "RF_TEST1.csv", quote = F, sep = ",")



# MODELO: RF - TEST2 - 14 variables (4 features nuevas) - NA = 0, inf = 100
rf.model.test1.part1 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100) 
rf.model.test1.part2 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100)
rf.model.test1.part3 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100)
rf.model.test1.part4 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100)
rf.model.test1.part5 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100)
rf.model.test1.all <- combine(rf.model.test1.part1,rf.model.test1.part2,rf.model.test1.part3,rf.model.test1.part4,rf.model.test1.part5)

pred.rf <- predict(rf.model.test1.all, test.data[,3:16], type = "prob")

write.table(pred.rf[,2], file = "RF_TEST2.csv", quote = F, sep = ",")



# MODELO: RF - TEST3 - 14 variables (4 features nuevas) - con mtry - NA = 0, inf = 100
rf.model.test4.part1 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test4.part2 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test4.part3 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test4.part4 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test4.part5 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test4.all <- combine(rf.model.test4.part1,rf.model.test4.part2,rf.model.test4.part3,rf.model.test4.part4,rf.model.test4.part5)

pred.rf <- predict(rf.model.test4.all, test.data[,3:16], type = "prob") 

write.table(pred.rf[,2], file = "RF_TEST3.csv", quote = F, sep = ",")


# MODELO: RF - TEST4 - 14 variables (4 features nuevas) - con mtry - NA = 0, inf = 50
rf.model.test2.part1 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test2.part2 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test2.part3 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test2.part4 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test2.part5 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test2.all <- combine(rf.model.test2.part1,rf.model.test2.part2,rf.model.test2.part3,rf.model.test2.part4,rf.model.test2.part5)

pred.rf <- predict(rf.model.test2.all, test.data[,3:16], type = "prob") 

write.table(pred.rf[,2], file = "RF_TEST4.csv", quote = F, sep = ",")



# MODELO: RF - TEST5 - 14 variables (4 features nuevas) - con mtry - NA = -10, inf = 50
rf.model.test5.part1 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test5.part2 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test5.part3 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test5.part4 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test5.part5 <- randomForest(train.data[,3:16], y = train.class, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test5.all <- combine(rf.model.test5.part1,rf.model.test5.part2,rf.model.test5.part3,rf.model.test5.part4,rf.model.test5.part5)

pred.rf <- predict(rf.model.test5.all, test.data[,3:16], type = "prob")

write.table(pred.rf[,2], file = "RF_TEST5.csv", quote = F, sep = ",")


# De todos los modelos probados, el que mejor resultado ha obtenido (aunque sin ser mucha diferencia) 
# es el Test5. Posiblemente sea por considerar los valores NA como -10 en vez de 0.