# En este archivo vamos preparar los datos que obtenemos en su formato comprimido así como convertir las variables que sean necesarias
# en factores. Además, usaremos SMOTE y undersampling.

library(tidyverse)
library(reshape2)
require(randomForest)

#Definimos esta opción para poder usar 14 gigas de memoria en las librerías que usen la máquina virtual de java.
options(java.parameters = "-Xmx14g")


#Usamos esta función para poder calcular el score F1 en cross validation
micro_averaged_f1_score <- function(true, pred){
  
  # Construimos la matriz de confusión
  confusion_table <- table(Actual = true, Predicted = pred)
  
  if(nrow(confusion_table) != ncol(confusion_table)){
    
    missings <- setdiff(rownames(confusion_table), colnames(confusion_table))
    
    missing_mat <- matrix(0, nc = length(missings), nr = nrow(confusion_table))
    colnames(missing_mat) <- missings
    
    confusion_table  <- as.table(cbind(as.matrix(confusion_table), missing_mat))
    confusion_table <- confusion_table[,rownames(confusion_table)]
  }
  
  confusion_matrix <- as.matrix(confusion_table)
  
  n <- sum(confusion_matrix) # Número de instancias
  nc <- nrow(confusion_matrix) # Número de clases
  
  diag <- diag(confusion_matrix) # Número de instancias bien clasificadas
  
  rowsums <- apply(confusion_matrix, 1, sum) # Número de instancias por clase
  colsums <- apply(confusion_matrix, 2, sum) # Número de predicciones por clase
  
  # Construimos nc matrices de confusión one-vs-all
  oneVsAll <- lapply(1:nc, function(i){
    v <- c(confusion_matrix[i,i],
           rowsums[i] - confusion_matrix[i,i],
           colsums[i] - confusion_matrix[i,i],
           n-rowsums[i] - colsums[i] + confusion_matrix[i,i])
    return(matrix(v, nrow = 2, byrow = T))})
  
  # Construimos una matriz sumando los valores de las nc anteriores
  s <- matrix(0, nrow = 2, ncol = 2)
  for(i in 1:nc){
    s <- s + oneVsAll[[i]]
  }
  
  # Calculamos la precision y recall para obtener el F1 score
  #precision <- diag(s) / apply(s, 2, sum) # TP / TP+FP
  #recall <- diag(s) / apply(s, 1, sum)    # TP / TP+FN
  #micro_f1 <- 2 * precision * recall / (precision + recall)
  
  # Dado que la suma por columnas y por filas da el mismo resultado en la matriz
  # s, podemos calcular el F1 score de la siguiente forma y obtenemos el mismo
  # resultado
  micro_f1 <- (diag(s) / apply(s, 1, sum))[1]
  
  return(micro_f1)
}

# Usamos esta función para poder generar el archivo submission.
create_submission <- function(y_pred_test, name){
  
  filename <- paste(name, "submission.csv", sep = "_")
  submission <- read.csv("submission_format.csv")
  submission[,2] <- y_pred_test
  write.csv(submission, filename, row.names = FALSE, quote = FALSE)
}



set.seed(123)
# Leemos los datos
train_x <- read.csv('train_values.csv')
train_y <- read.csv('train_labels.csv')
test_x <- read.csv('test_values.csv')

# Eliminamos la columna building_id, ya que es un identificador único
train_x <- train_x[,-1]
train_y <- train_y[,-1, drop = FALSE]
test_x <- test_x[,-1]
train <- cbind(train_x, train_y)

#no necesitamos esta variables
rm(train_x, train_y)

#Estudiamos las variables con un str()
str(train)



#Como vemos, hay variables en formato int o character que sabemos que tendrían que ser factores.
#Las convertimos.

train = mutate_if(train, is.character, as.factor)
train$damage_grade = factor(train$damage_grade)
test_x = mutate_if(test_x, is.character, as.factor)


#A continuación vamos a producir los datos shrunken. Como vemos, hay dos conjuntos de variables en formato one-hot
#que corresponden al tipo de material utilizado en la construcción así como el uso secundario que tiene el edificio en cuestión.
#Una posible idea era condensar estas variables como variables factores ya que esperaba que los árboles, al menos J48, prefiera
#variables que tengan varias salidas de factores.

#######################################################
prueba = train[, 15:25]
sapply(1:length(colnames(prueba)), function(x){
  substring(colnames(prueba)[x], 20, nchar(colnames(prueba)[x]))
})
colnames(prueba) = sapply(1:length(colnames(prueba)), function(x){
  substring(colnames(prueba)[x], 20, nchar(colnames(prueba)[x]))
})



#El problema de esto es que tenemos muchos casos en los que varias columnas de estas one hot encoding tienen un valor 1:

p = apply(prueba, 1, sum)
sum(p != 1)
rm(p)

#Obtenemos 83585 instancias, son muchas. Aún así, podemos probar cómo afecta al rendimiento aunque perdamos información.
#Además también puede servir para selección de instancias o búsquedas de patrones.
##########################


prueba = data.frame(type = names(prueba)[max.col(prueba)])

train.long = train[, -c(15:25)]
train.long = data.frame(train.long, material = prueba)
train.long$type = factor(train.long$type)

aux1 = train[, 28]
aux2 = train[, 29:38]
aux2.names = sapply(1:length(colnames(aux2)), function(x){
  substring(colnames(aux2)[x], 19, nchar(colnames(aux2)[x]))
})
colnames(aux2) = aux2.names
secondary_use = ifelse(aux1 != 0, names(aux2)[max.col(aux2)], "No")
train.long = train.long[, -c(17:27)]
train.long = data.frame(train.long, secondary_use = secondary_use)
train.long$secondary_use = factor(train.long$secondary_use)
rm(aux1, aux2, prueba, secondary_use)
#######################################################

#Repetimos lo mismo con test


prueba = test_x[, 15:25]
sapply(1:length(colnames(prueba)), function(x){
  substring(colnames(prueba)[x], 20, nchar(colnames(prueba)[x]))
})
colnames(prueba) = sapply(1:length(colnames(prueba)), function(x){
  substring(colnames(prueba)[x], 20, nchar(colnames(prueba)[x]))
})
prueba = data.frame(type = names(prueba)[max.col(prueba)])
test_x.long = test_x[, -c(15:25)]
test_x.long = data.frame(test_x.long, material = prueba)
test_x.long$type = factor(test_x.long$type)


aux1 = test_x[, 28]
aux2 = test_x[, 29:38]
aux2.names = sapply(1:length(colnames(aux2)), function(x){
  substring(colnames(aux2)[x], 19, nchar(colnames(aux2)[x]))
})
colnames(aux2) = aux2.names
secondary_use = ifelse(aux1 != 0, names(aux2)[max.col(aux2)], "No")
test_x.long = test_x.long[, -c(17:27)]
test_x.long = data.frame(test_x.long, secondary_use = secondary_use)
test_x.long$secondary_use = factor(test_x.long$secondary_use)
rm(aux1, aux2, prueba, secondary_use, aux2.names)

#########################################################
#De momento tenemos la variante raw de nuestros datos: train y test_x 
#así como la versión comprimida: train.long y test_x.long



#Antes de comenzar con el eda en el otro script, vamos a obtener aquí los datos 
#con undersampling y SMOTE

#Queremos recordar que esta no es la forma habitual de proceder, primer se hacer un eda
#con los datos originales y conforme existan distinta necesidad de convertir los datos
#se procede a aplicar distintos algoritmos, así fue la forma en la que se ha procedido
#en la realización de la práctica.
#Pero, de cara a la presentación, es más cómodo meter en un mismo script toda la conversión
#de datos. 


#UNDERSAMPLING
set.seed(1)
#Usaremos la librería UBL
library(UBL)

#Creamos los datos para train y test:
train.long.balanced = RandUnderClassif(damage_grade~., train.long, "balance")
train.balanced = RandUnderClassif(damage_grade~., train, "balance")

#El hecho de usar la opción balance, como veremos, permite que haya la misma proporción de clases.

#SMOTE
set.seed(1)
#Usamos la librería:
library(DMwR)

#Como aplicación de una técnica vista en la asignatura: minería de datos, aspectos avanzados; utilizamos
#la técnica SMOTE que implementa la generación de nuevos casos de la clase minoritaria así como un pequeño
#undersampling de la mayoritaria.
train.long.smote = SMOTE(damage_grade ~., train.long)
train.smote = SMOTE(damage_grade ~., train)

######################################################################################################


