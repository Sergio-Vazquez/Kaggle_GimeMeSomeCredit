## PRACTICA APRENDIZAJE SUPERVISADO ##
############ OPCION 2 - PARTE 2  - MODELO DE REGRESION ################

# Primero, descargamos los datos de kaggle: https://www.kaggle.com/c/GiveMeSomeCredit

# En esta segunda partede la practica, el objetivo el predecir el sueldo mensual de la persona.


######### MODELO DE REGRESION PARA PREDECIR EL SUELDO MENSUAL ############

# Cargamos las librerias:
library(randomForest)
library(caret)
library(tree)
library(C50)

# Creamos los conjuntos train y test
train.data <- read.csv("src/cs-training.csv")
test.data <- read.csv("src/cs-test.csv")
submissision <- read.csv("src/sampleEntry.csv")


# Limpiamos los datos con valor NA
train.data[is.na(train.data[,12]), 12] <- -10  # aqui le doy -10 a las variables NA de "NumberOfDependences"
train.data[is.na(train.data[,7]), 7] <- -10 # aqui le doy -10 a las variables NA de "MonthlyIncome"

test.data[is.na(test.data[,12]), 12] <- -10  # aaqui le doy -10 a las variables NA de "NumberOfDependences"
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



### Calculamos la regresion:

### MODELO REGRESION TREE ###

##### PAQUETE TREE #####
set.seed(1)
library(tree)

train <- sample(1:nrow(train.data), nrow(train.data)/2)

# Creamos el modelo con la limitacion de subset:
tree.Monthly <- tree(MonthlyIncome ~., train.data, subset = train)

# Probamos un modelo con todos los datos, sin subset:
tree.Monthly2 <- tree(MonthlyIncome ~., train.data)

summary(tree.Monthly)

plot(tree.Monthly)
text(tree.Monthly, pretty = 0)
# se puede ver que utiliza 5 variables

# usemos cvtree() para ver si podando el arbol se mejora. 
cv.Monthly <- cv.tree(tree.Monthly)
plot(cv.Monthly$size, cv.Monthly$dev, type = 'b')


# Hacemos lo mismo que antes, pero con todos los datos, para ver mejor la tendencia: 
cv.Monthly2 <- cv.tree(tree.Monthly2)
plot(cv.Monthly2$size, cv.Monthly2$dev, type = 'b') # se ve como a partir de size = 6 empieza a decaer

# Podamos el arbol a partir de lo obtenido en la grafica anterior:
prune.Monthly <- prune.tree(tree.Monthly, best = 6)
plot(prune.Monthly)
text(prune.Monthly, pretty = 0)

# Hacemos una prediccion con el arbol sin podar
yhat <- predict(tree.Monthly, newdata = train.data[-train, ])
Montly.test <- train.data[-train, "MonthlyIncome"]
plot(yhat, Montly.test)  # en el eje y es el valor real y en el eje x el valor predicho.
abline(0, 1) # obtenemos la regresion
mean((yhat - Montly.test)^2)

# Ahora hacemos lo mismo, pero con el arbol podado.
yhat2 <- predict(prune.Monthly, newdata = train.data[-train, ])
Montly.test2 <- train.data[-train, "MonthlyIncome"]
plot(yhat2, Montly.test2)  # en el eje y es el valor real y en el eje x el valor predicho.
abline(0, 1) # obtenemos la regresion
mean((yhat2 - Montly.test2)^2)

# En este caso, obtenemos peores resultados con el arbol podado que sin podar.


#### CON PAQUETE RPART #####

# Instalamos el paquete de R
library(rpart)

# Creamos un arbol de decision para predecir la variable MonthlyIncome utilizando todas las demas.

# cp estable la profundidad del arbol
rpart.tree <- rpart(MonthlyIncome ~ ., data = train.data, cp = 10^(-6))

# Obtenemos los nombres de la variable anterior
names(rpart.tree)

# Informacion del tamaño del arbol y el error:
rpart.tree$cptable[1:10, ]

# Error de los ultimos arboles:
rpart.tree$cptable[dim(rpart.tree$cptable)[1] - 13:0, ]

# El ultimo arbol tiene 615 splits. Vamos a limitarlos a 9
cp9 <- which(rpart.tree$cptable[, 2] == 9)
rpart.tree.9 <- prune(rpart.tree, rpart.tree$cptable[cp9, 1])

# Mostramos el arbol con print() y summary()
print(rpart.tree.9)
summary(rpart.tree.9)

# Hacemos una representacion grafica del arbol y lo guardamos en png:
png("rpart.tree.9.png", width = 1200, height = 800)
post(rpart.tree.9, file = "", title = "Clasificando las donaciones con un arbol
     de profundidad 9", bp = 18)
dev.off()

# Vamos a mejorar el plot anterior con los paquetes rattle y rpart.plot:
library(rattle)
library(rpart.plot)

prp(rpart.tree.9)  
fancyRpartPlot(rpart.tree.9)

