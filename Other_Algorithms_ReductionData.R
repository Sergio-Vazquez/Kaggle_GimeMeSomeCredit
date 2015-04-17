## PRACTICA APRENDIZAJE SUPERVISADO ##
############ OPCION 2 - PARTE 1 - PRUEBA VARIOS MODELOS CON LIMITACION DATOS ################

# Primero, descargamos los datos de kaggle: https://www.kaggle.com/c/GiveMeSomeCredit

# El objetivo es predecir si el cliente tendra problemas financieros en los proximos 2 años.
# La variable "SeriousDlqin2yrs" sera la variable a predecir



######### PRUEBA DE DISTINTOS MODELOS CON REDUCCION DE DATOS PARA PODER PROCESAR ############

# Cargamos las librerias:
library(randomForest)
library(caret)
library(tree)
library(C50)

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

# Limpiamos los datos con valor NA
train.data[is.na(train.data[,12]), 12] <- -10  # aqui le doy -10 a las variables NA de "NumberOfDependences"
train.data[is.na(train.data[,7]), 7] <- -10 # aqui le doy -10 a las variables NA de "MonthlyIncome"

test.data[is.na(test.data[,12]), 12] <- -10  # aqui le doy -10 a las variables NA de "NumberOfDependences"
test.data[is.na(test.data[,7]), 7] <- -10 # aqui le doy -10 a las variables NA de "MonthlyIncome"

#### CREACION DE NUEVAS FEATURES ######

# PrestamosIngresos = NumberOfOpenCreditLinesAndLoans/MonthlyIncome
train.data$PrestamosIngresos <- (train.data$NumberOfOpenCreditLinesAndLoans/train.data$MonthlyIncome)
test.data$PrestamosIngresos <- (test.data$NumberOfOpenCreditLinesAndLoans/test.data$MonthlyIncome)
# Elimino valores NA y asigno valor a inf
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

########## LIMITACION DE LOS DATOS ###############

# Creo un nuevo conjunto de training con menos observaciones para poder trabajar con mi PC:
# Lo limitamos a un 10% (p=0.10) o un 2% (p=0.02):
set.seed(1234)
intrain <- createDataPartition(train.data$SeriousDlqin2yrs, times=1, p=0.02, list=F)

train.v2<-train.data[intrain,]
train.class.v2 <- train.v2[["SeriousDlqin2yrs"]]


####### MODELO RANDOM FOREST CON 500 TREES, 10% DATOS  ####### 

rf.model.test6.part1 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14)) # class es el metodo de random forest, 
rf.model.test6.part2 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14)) # que utiliza el metodo de clasificacion.
rf.model.test6.part3 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test6.part4 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test6.part5 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test6.all <- combine(rf.model.test6.part1,rf.model.test6.part2,rf.model.test6.part3,rf.model.test6.part4,rf.model.test6.part5)

pred.rf <- predict(rf.model.test6.all, test.data[,3:16], type = "prob") # predecimos en el modelo de test dado por kaggle el modelo anterior

write.table(pred.rf[,2], file = "RF_TEST6_10%.csv", quote = F, sep = ",")  


####### MODELO RANDOM FOREST CON 500 TREES, 2% DATOS  ####### 

rf.model.test7.part1 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14)) # class es el metodo de random forest, 
rf.model.test7.part2 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14)) # que utiliza el metodo de clasificacion.
rf.model.test7.part3 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test7.part4 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test7.part5 <- randomForest(train.v2[,3:16], y = train.class.v2, method = "class",ntree = 100, mtry =  sqrt(14))
rf.model.test7.all <- combine(rf.model.test7.part1,rf.model.test7.part2,rf.model.test7.part3,rf.model.test7.part4,rf.model.test7.part5)

pred.rf <- predict(rf.model.test7.all, test.data[,3:16], type = "prob") # predecimos en el modelo de test dado por kaggle el modelo anterior

write.table(pred.rf[,2], file = "RF_TEST7_2%.csv", quote = F, sep = ",")


## Se obtienen mejores resultados con la muestra del 10%, pero la mejora no es considerable.


####### MODELO DECISION TREES, C5.0, 2% DATOS  ####### 
cvControlDT <- trainControl(number = 10, method = "repeatedcv", repeats = 10)


c50.fit2 <-  train(train.v2[,3:16], train.class.v2, method = "C5.0", tuneLength = 10, metric = "Kappa",
                  trControl = cvControlDT)
# Prediccion
pred.c502 <- predict(c50.fit2$finalModel, newdata = test.data[,3:16], type = "prob")
write.table(pred.c502[,2], file = "DT_TEST1_2%.csv", quote = F, sep = ",")


####### MODELO NEURAL NETWORK, NNET, 2% DATOS ####### 

cvControlNN <-  trainControl(number = 5, method = "repeatedcv", repeats = 5)

nn.fit <- train(train.v2[,3:16], train.class.v2, method = "nnet", tuneLength = 5, metric = "Kappa",
                trControl = cvControlNN)

# Predecimos:
pred.nn <- predict(nn.fit$finalModel, newdata = test.data[,3:16], type = "raw")
write.table(pred.nn[,1], file = "NN_TEST1_2%.csv", quote = F, sep = ",")


# Decision Tree obtiene buenos resultados para ser solo un 2% de los datos.
# Por otro lado, el modelo Neural Network, obtiene unos pesimos resultados con el mismo porcentaje de datos.
# El modelo Random Forest, fue el mas rapido y el que  mejor resultado dio, incluso con el 100 % de los datos, al tener la
# posibilidad de realizar varios trees y combinarlos, el tiempo de ejecucion fue mucho mejor que incluso el 2% del modelo NN.
