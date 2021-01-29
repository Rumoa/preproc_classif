
library(DMwR)
library(UBL)
library(tidyverse)
library(reshape2)
library(randomForest)
library(caret)


#Para cargar los datos podemos usar el fichero conversion.R,
#CUIDADO, al cargarlo ejecutamos SMOTE y puede tardar mucho, recomendamos 
#cargar el workspace after_conversion.RData
source("conversion.R")

setwd("files")
load("after_conversion.RData")

#Tenemos los datos originales en train, en primer lugar, comprobamos que no haya missing values:
sum(is.na(train))

#Podemos hacer un str de los datos y ver que en el script conversion.R hemos convertido correctamente
#los tipos de las variables en factor cuando sea necesario
str(train)

#Estudiamos la proporción entre las clases de la variable objetivo:
prop.table(table(train$damage_grade)) %>% round(2)

#Como vemos hay un desbalanceo importante, con sólo un 10% de la clase 1 y un 33% de la clase 3.
#Por ello, parece tener sentido usar undersampling (que elimina instancias de las clases mayoritarias) 
#así como SMOTE para conseguir nuevos casos de la clase minoritaria (aunque también undersamplea de la 
#mayoritaria)


#En el script anterior hemos procedido a usar undersampling y SMOTE y los hemos guardado en las variables
#train.balanced y train.smote (y en sus versiones de formato comprimidas, ver más adelante)

#Veamos la proporción:
#Undersampling:
prop.table(table(train.balanced$damage_grade)) %>% round(2)
#Obtenemos una misma proporción entre las clases, sin embargo:
dim(train.balanced)
#hay sólo 75372 instancias, en vez de las 260601 originales.
#A priori podemos pensar que la clasificación debe mejorar al evitar el desbalanceo:
#No es así, ya que perdemos muchos datos.


#SMOTE:
prop.table(table(train.smote$damage_grade)) %>% round(2)
#Aquí sigue habiendo un desbalanceo pero menor, en concreto, la clase mayoritaria es 1 
#y la minoritaria ahora es 3.
#Es posible que esto se deba a que SMOTE funcione mejor con clasificación binaria. Aunque no 
#estamos seguros

#No obstante, vemos que:
dim(train.smote)
#Ahora tenemos 175868, que corresponde a un 67% de los datos originales.
#Aunque haya desbalanceo, podemos esperar que la clasificación funcione mejor con este que con
#undersampling, simplemente por la mayor cantidad de datos.


#FORMATO Shrunken.
#Tal y como hemos explicado en el script anterior, hay dos conjuntos de variables correspondientes
#al material usado en la construcción así como en si tiene uso secundario o no; que aparecen en 
#formato one-hot encoding.
#
#Pensando que los árboles se pueden beneficiar de la mayor cantidad de factores en variables, 
#intentamos comprimir los datos de estos dos conjuntos de variables en dos variables ``type" y
#``secondary_use" en la que el factor de esa celda sea el correspondiente al uno de su familia
#de variables.
#El problema es que hay 83585 instancias con varios valores de 1 dentro de una misma familia,
#por lo que estamos creando datos falsos. Aún así, nos puede ayudar a ver algún tipo de patrón en 
#en los datos así como poder usar selección de características.

#La conversión ya se ha realizado en el script anterior y se llama train.long (y en sus versiones \
#smote y unbalanced)

#Vemos con str cómo han quedado los datos
str(train.long)
#Como vemos ahora hay dos variables type y secondary_use que condensan (no de forma completamente real) 
#la información de las dos familias de variables. 

#Aún así, probaremos los algoritmos en cada conjunto de datos.


#Búsqueda de patrones interesantes:

#Haciendo uso de los datos originales en forma comprimido, podemos buscar cómo se distribuyen los 
#valores de las variables en función del daño que hayan recibido los edificios:


dmg.1 = train.long %>% filter(damage_grade==1)
dmg.2 = train.long %>% filter(damage_grade==2)
dmg.3 = train.long %>% filter(damage_grade==3)

#definimos esta función para ver qué valores salen con más frecuencia:
see.freq =  function(col, data){
  l = prop.table(table(data[,col])) %>% round(digits = 4)
  l = data.frame(l)
  colnames(l)[1] = colnames(data)[col]
  l %>% arrange(desc(Freq))
}



a=lapply( 4:length(dmg.1), see.freq, dmg.1)
b=lapply( 4:length(dmg.2), see.freq, dmg.2)
c=lapply( 4:length(dmg.3), see.freq, dmg.3)

#Mostramos los resultados
for (i in 1:length(a)){
  print('dmg.1')
  print(a[[i]])
  print('')
  print('dmg.2')
  print(b[[i]])
  print('')
  print('dmg.3')
  print(c[[i]])
  print('###########')
}

#Podemos extraer la siguiente información:
#Foundation type: en las clases 2 y 3, la gran mayoría está en r, mientras que en la 1 está más repartida
#other floor type: en las clases 2 y 3, la gran mayoría es q, mientras que en 1 la mayoría es j
#roof type tiene en las clases 2 y 3 un mayor porcentaje n, que en la clase 1.
#Ground floor type: en las clases 2 y 3 la mayoría es f y en la 1 se reparte más.
#Position: tiene la mayoría s en las tres clases

#En cuento a los materiales, las que más sufren daño (dmg 3), un 74% son las de mortar stone, mientras
#que en dmg1 son un 26% solo
#En cuanto al uso secundario: la mayoría no lo tienen, pero las que sí:
#* los que menos daño tienen son los hoteles
#* y damage 2 y 3 las destinadas a la agricultura
#
#En las edades también hay desbalanceo, ya que las que tiene menos daño tienen poca edad
#mientras que las que sufren más daño, alrededor de un 15% para dmg2 y dmg3 tienen 15 años.




###################################################################
#Estudiamos la correlación que existe entre las variables que tenemos:
#seleccionamos las variables numéricas (únicamente 8)
library(corrplot)
col.num = unlist(lapply(train.long, is.numeric))
corrplot(cor(train.long[col.num]))

#Vemos que existe una correlación grande entre height_percentage y count_floors_pre_eq.
#Podemos eliminarla cuando hagamos distintos modelos, aunque preferimos ver primero
#la importancia que tienen las variables con FSelector

###########################################################
#Selección de variables.

#Ya que vamos a usar árboles de decisión, podemos usar Fselector, que permite saber
#la importancia de las variables en el trabajo de clasificación. Para ello crea distintos
#árboles de decisión usando random forest, por lo que, de forma indirecta, usamos el 
#algoritmo restante de la asignatura. Para ello usaremos Fselector junto a los datos
#comprimidos que permiten ver la información de forma más compacta

library(FSelector)
#CUIDADO ESTO PUEDE TARDAR MUCHO TIEMPO, ejecutar con cuidado
att.scores = random.forest.importance(damage_grade ~., train.long)
#saveRDS(att.scores, "att.scores.rds")
#Podemos cargarlo directamente de la carpeta
att.scores = readRDS("att.scores.rds")
cutoff.biggest.diff(att.scores)
#Vemos que la variables más importantes son los identificadores de la zona geográfica
#Ordenamos los atributos en forma descendiente:
data.frame(att.scores) %>% arrange(desc(attr_importance))

#En caso de intentar mejorar nuestros modelos, podemos probar quitando variables.
#En nuestro caso, quitaremos las dos últimas, ya que empeora al quitar legal_ownership_status.



##################################################################
#Generación de modelos.

#La estrategia que seguiremos es la siguiente, proponemos varios modelos que usarán variaciones
#de los datos con todas las variables y, seleccionando el que mejor puntuación obtenga con el conjunto
#test de la página, quitaremos variables con el fin de mejorar la puntuación.
#En el caso ganador, intentaremos tunear los posibles parámetros que contenga el algoritmo.

#En nuestro caso usaremos las funciones tree y J48. Sabemos que son algoritmos distintos y esperamos
#un mejor rendimiento de J48.

#Para cada modelo guardaremos un archivo RDS que lo contenga así como posible información de interés (f1, etc)
#para poder cargarlo y despejar la memoria de R en caso de que así se vea necesario.

#Si el lector desea comprobar los resultados sin tener que ejecutar todos los algoritmos (pueden tardar varias
#horas), recomendamos acceder a este repositorio donde se pueden encontrar los archivos:


# https://github.com/Rumoa/preproc_classif

#No recomendamos ejecutar todos los comandos ya que se puede demorar la espera.


#Definimos cuatro funciones que nos permite automatizar el proceso.

#Esta función permite entrenar un modelo de j48 con la fórmula que nos interese,
#hacer predicciones con un set de test, su score f1 y guardarlo en un archivo RDS par
#poder utilizarlo
j48train = function(data, test, formula, nombre, export = "yes"){
  set.seed(1)
  require(RWeka)
  require(rJava)
  
  model = J48(formula, data)
  model.predict = predict(model, newdata = data)
  
  micro_f1 = micro_averaged_f1_score(model.predict , data$damage_grade)
  
  model.predict.test = predict(model, newdata=test)
  
  if (export == "yes"){
    create_submission(model.predict.test,  paste("j48.",nombre, sep = ""))
    .jcache(model$classifier)
    saveRDS(model, paste("j48.",nombre, ".rds", sep=""))
  }
  l = list( modelo = model,
            prediccion.train = model.predict,
            micro_f1.train = micro_f1,
            prediccion.test = model.predict.test
  )
  return(l)
}

#Uno de los problemas que tiene esto es que no podemos usar la función 
#evaluate_Weka_classifier cuando cargamos un RDS, ya que da error.
#Tampoco la opción summary.
#No ocurre si ejecutamos el algoritmo y directamente se guarda en memoria.

#Tenemos que crear una función que nos permita hacer cross validation.

#Definimos esta función para poder calcular la media del score f1 de acuerdo 
#al formato en el que vamos a guardar los modelos en la función siguiente de
#cross validation
media_f1.cv = function(aux){ 
  mean = 0
  for (i in 1:length(aux)){
    mean = aux[[1]]$f1 + mean
  }
  mean = mean/length(aux)
  return(mean)
}


#Función para hacer cross validation con j48, permite exportarlos como un archivo .RDS.
#Se calculan nfolds modelos para los cuales se calculan f1, la predicción con el test que toque
#en ese fold y la tabla de confusión
cvJ48 =  function(data, formula, nombre, nfolds =5, export = "yes"){ 
  require(RWeka)
  require(rJava)
  require(caret)
  set.seed(1)
  folds.balanced <- createFolds(data$damage_grade, k = nfolds)
  str(folds.balanced)
  
  
  result = lapply(1:length(folds.balanced), function(x){
    tr = folds.balanced[-x]
    tr = unlist(tr)
    tst = folds.balanced[x]
    tst = unlist(tst)
    
    model = J48(formula, train.balanced[tr,])
    .jcache(model$classifier)
    model.pred = predict(model, newdata = train.balanced[tst,])
    f1 = micro_averaged_f1_score(model.pred , train.balanced[tst,]$damage_grade)
    tabla = table(model.pred, train.balanced[tst,]$damage_grade)
    
    
    
    
    return( list(modelo = model,
                 prediccion = model.pred,
                 f1 = f1,
                 confusion.matrix = tabla
    )
    )
    
    
    
  })
  
  if (export == "yes"){
    saveRDS(result, paste("j48.cv.", nombre, ".rds", sep=""))
  }
  return(result)
}



#Con la función tree repetimos de forma similar:
TREE.train = function(data, test, formula, nombre){
  set.seed(1)
  require(tree)
  
  model = tree(formula, data)
  model.predict = predict(model, newdata = data, type ="class")
  
  micro_f1 = micro_averaged_f1_score(model.predict , data$damage_grade)
  
  model.predict.test = predict(model, newdata=test, type ="class")
  create_submission(model.predict.test, paste("TREE.",nombre, sep="") )
  saveRDS(model, paste("TREE.", nombre, ".rds", sep=""))
  
  l = list( modelo = model,
            prediccion.train = model.predict,
            micro_f1.train = micro_f1,
            prediccion.test = model.predict.test
  )
}

#Y también implementamos su versión para cross validation.
cvtree =  function(data, formula, nombre, nfolds =5, export = "yes"){ 
  require(tree)
  set.seed(1)
  folds.balanced <- createFolds(data$damage_grade, k = nfolds)
  str(folds.balanced)
  
  
  result = lapply(1:length(folds.balanced), function(x){
    tr = folds.balanced[-x]
    tr = unlist(tr)
    tst = folds.balanced[x]
    tst = unlist(tst)
    
    model = tree(formula, train.balanced[tr,])
    model.pred = predict(model, newdata = train.balanced[tst,], type ="class")
    f1 = micro_averaged_f1_score(model.pred , train.balanced[tst,]$damage_grade)
    tabla = table(model.pred, train.balanced[tst,]$damage_grade)
    
    
    
    
    return( list(modelo = model,
                 prediccion = model.pred,
                 f1 = f1,
                 confusion.matrix = tabla
    )
    )
    
    
    
  })
  
  if (export == "yes"){
    saveRDS(list(result = result, media_f1 = media_f1.cv(result)), paste("TREE.cv", nombre, ".rds", sep=""))
  }
  return(result)
}



########################################################################################
#TREE

#En primer lugar, utilizamos la librería tree y su función tree para producir un modelo que nos 
#indique cómo se comporta esta función con los datos.
library(tree)
set.seed(1)
tree.prueba = tree(damage_grade ~., train)
#predecimos con el conjunto de entrenamiento
tree.prueba.predict = predict(tree.prueba, newdata = train, type = "class")
#Veamos el score f1 obtenido:
micro_averaged_f1_score(tree.prueba.predict, train$damage_grade)
#Obtenemos un valor de 0.64, no es muy alto.

#Veamos un summary del árbol:
summary(tree.prueba)
#Tiene 9 hojas. El problema radica en que sólo dos variables se usaron en 
#la construcción del árbol.

#Veámoslas
plot(tree.prueba)
text(tree.prueba)

#Sólo las variables foundation_type y geo_level_1_id
#se utilizaron, lo cual es molesto ya que no estamos utilizando toda la información
#disponible.

#Procedemos a podar el árbol para ver si conseguimos alguna mejora:

#
cv.prune = cv.tree(tree.prueba, FUN=prune.misclass)
cv.prune
#Vemos que la que mons desviación tiene son los árboles que tienen 9 y 6 hojas.

#Podemos ver la predicción con conjunto train para ver si conseguimos alguna mejoría con 6 hojas.
tree.pruned = prune.misclass(tree.prueba, best=6)


cv.pruned.predict = predict(tree.pruned, newdata = train, type ="class")
micro_averaged_f1_score(cv.pruned.predict, train$damage_grade)

#Obtenemos la misma clasificación. Además, como no consume demasiada memoria, nos da igual 
#quedarnos con el árbol de 6 o 9 hojas ya que ambos producen una clasificación ``mala".

#Distintas pruebas durante la realización del trabajo han mostrado que no podemos podar bien
#en este conjunto de datos. Por ejemplo, con formato comprimido:


tree.prueba = tree(damage_grade ~., train.long)
tree.prueba.predict = predict(tree.prueba, newdata = train.long, type = "class")
micro_averaged_f1_score(tree.prueba.predict, train$damage_grade)
#Obtenemos el mismo valor, tiene sentido ya que las variables que cambiaban no 
#se han utilizado en la realización del árbol.

#De todas formas, es mejor usar cross validation para estimar el rendimiento del algoritmo
#tree:

#Formato original
tree.cv.wide.all = cvtree(train, damage_grade~., "wide.all", nfolds =5)
gc()
#Formato original set balanced
tree.cv.wide.balanced = cvtree(train.balanced, damage_grade~., "wide.balanced", nfolds =5)
gc()
#Formato original con SMOTE
tree.cv.wide.smote = cvtree(train.smote, damage_grade~., "wide.smote", nfolds =5)
gc()
#Probamos a quitar la variable de menos importancia de Fselector
tree.cv.wide.count = cvtree(train, damage_grade ~. -count_families, "wide.count", nfolds =5)
gc()
#Además de quitar la variable de menor importancia, también quitamos la siguiente en orden 
#de importancia.
tree.cv.wide.count.plan = cvtree(train, damage_grade ~. -count_families -plan_configuration, "wide.count.plan", nfolds =5)
gc()

#Podemos importar los archivos obtenidos del repositorio.


###########Formato shrunken#################
#Shrunken con todas las variables
tree.cv.long.all = cvtree(train.long, damage_grade~., "long.all", nfolds =5)
gc()
#Shrunken con balanced
tree.cv.long.balanced = cvtree(train.long.balanced, damage_grade~., "long.balanced", nfolds =5)
gc()
#Shrunken con SMOTE
tree.cv.long.smote = cvtree(train.long.smote, damage_grade~., "long.smote", nfolds =5)
gc()
#Shrunken quitando la variable count_families
tree.cv.long.count = cvtree(train.long, damage_grade ~. -count_families, "long.count", nfolds =5)
gc()
#Shrunken quitando la variable count_families y la variable plan_configuration
tree.cv.long.count.plan = cvtree(train.long, damage_grade ~. -count_families -plan_configuration, "long.count.plan", nfolds =5)
gc()



##########################################
#En caso de quere cargar los archivos
ficheros.tree.cv = list( "TREE.cvlong.all.rds"       , "TREE.cvlong.balanced.rds"   ,"TREE.cvlong.count.plan.rds"
,"TREE.cvlong.count.rds"      ,"TREE.cvlong.smote.rds"      ,"TREE.cvwide.all.rds"       
,"TREE.cvwide.balanced.rds"   ,"TREE.cvwide.count.plan.rds" ,"TREE.cvwide.count.rds"     
,"TREE.cvwide.smote.rds")

ficheros.tree.cv = unlist(ficheros.tree.cv)
#Podemos importar los archivos obtenidos del repositorio:



tree.cv.wide.all = readRDS(ficheros.tree.cv[6])
tree.cv.wide.balanced =readRDS(ficheros.tree.cv[7])
tree.cv.wide.smote =readRDS(ficheros.tree.cv[10])
tree.cv.wide.count = readRDS(ficheros.tree.cv[9])
tree.cv.wide.count.plan =readRDS(ficheros.tree.cv[8])
tree.cv.long.all =readRDS(ficheros.tree.cv[1])
tree.cv.long.balanced = readRDS(ficheros.tree.cv[2])
tree.cv.long.smote = readRDS(ficheros.tree.cv[5])
tree.cv.long.count = readRDS(ficheros.tree.cv[4])
tree.cv.long.count.plan =readRDS(ficheros.tree.cv[3])



#Cada archivo es una lista que tiene una lista con los resultados del algortimo y el resultado 
#de hacer la media F1 (que es lo que nos interesa en este momento).

#Veamos la puntuación de cada método:
ficheros.cv.tree = ls(pattern = "tree.cv*")
ficheros.cv.tree = ficheros.cv.tree[-1]
l = list(0)
j = 0
for (i in ficheros.cv.tree){
  j= j +1
  aux = get(i)
  l[[j]] = aux$media_f1
}

cv.tree.results = data.frame(Method = ficheros.cv.tree, F1_Score = unlist(l))

#Veamos los resultados:
cv.tree.results

#Vemos que todos obtienen una puntuación muy parecida (menor que usando todo el conjunto de training),
#probablemente porque no se están usando apenas variables.


#No obstante, queremos probar el rendimiento de estos modelos con el conjunto de datos test, por lo
#que creamos los siguientes modelos y predecimos con el conjunto test:


#tree con todas las variables
tree.wide.all = TREE.train(train, test_x, damage_grade~., "wide.all")
gc()
#tree con formato ``ancho" balanceado
tree.wide.balanced = TREE.train(train.balanced, test_x, damage_grade~., "wide.balanced")
gc()
#tree con smote
tree.wide.smote = TREE.train(train.smote, test_x, damage_grade~., "wide.smote")
gc()
#tree con formato ancho sin usar la variable count_families
tree.wide.count = TREE.train(train, test_x, damage_grade ~. -count_families, "wide.count")
gc()
#tree con formato ancho sin usar la variable anterior ni plan_configuration
tree.wide.count.plan = TREE.train(train, test_x, damage_grade ~. -count_families -plan_configuration, "wide.count.plan")
gc()

###########Formato shrunken#################
#tree shrunken con todas las variables
tree.long.all = TREE.train(train.long, test_x.long, damage_grade~., "long.all")
gc()
#tree shrunken con datos balanced
tree.long.balanced = TREE.train(train.long.balanced, test_x.long, damage_grade~., "long.balanced")
gc()
#tree shrunken con smote
tree.long.smote = TREE.train(train.long.smote, test_x.long, damage_grade~., "long.smote")
gc()
#tree shrunken sin usar la variable count_families
tree.long.count = TREE.train(train.long, test_x.long, damage_grade ~. -count_families, "long.count")
gc()
#tree shrunken sin usar la variable anterior ni plan_configuration
tree.long.count.plan = TREE.train(train.long, test_x.long, damage_grade ~. -count_families -plan_configuration, "long.count.plan")
gc()

####################################################################################################
#Podemos cargarlo de nuevo si descargamos los archivos del repositorio y no queremos esperar a que se 
#generen los modelos
tree.models = list.files(pattern = "TREE.*")
tree.models = tree.models[-grep(tree.models, patter = "TREE.cv*")]
tree.models = tree.models[grep(tree.models, pattern = "\\.rds$")]


tree.wide.all = readRDS(tree.models[6])
tree.wide.balanced = readRDS(tree.models[7])
tree.wide.smote =readRDS(tree.models[10])
tree.wide.count =readRDS(tree.models[9])
tree.wide.count.plan = readRDS(tree.models[8])
tree.long.all =readRDS(tree.models[1])
tree.long.balanced = readRDS(tree.models[2])
tree.long.smote = readRDS(tree.models[5])
tree.long.count = readRDS(tree.models[4])
tree.long.count.plan = readRDS(tree.models[3])

#En estas variables ahora sólo están los árboles originales, en caso de querer predecir de nuevo
#los valores para el conjunto test, debemos:
require(tree)
tree.wide.all.predict_test = predict(tree.wide.all, newdata = test_x, type = "class")
tree.wide.balanced.predict_test = predict(tree.wide.balanced, newdata = test_x, type = "class")
tree.wide.smote.predict_test =predict(tree.wide.smote, newdata = test_x, type = "class")
tree.wide.count.predict_test =predict(tree.wide.count, newdata = test_x, type = "class")
tree.wide.count.plan.predict_test = predict(tree.wide.count.plan, newdata = test_x, type = "class")
tree.long.all.predict_test =predict(tree.long.all, newdata = test_x.long, type = "class")
tree.long.balanced.predict_test = predict(tree.long.balanced, newdata = test_x.long, type = "class")
tree.long.smote.predict_test =predict(tree.long.smote, newdata = test_x.long, type = "class")
tree.long.count.predict_test = predict(tree.long.count, newdata = test_x.long, type = "class")
tree.long.count.plan.predict_test = predict(tree.long.count.plan, newdata = test_x.long, type = "class")


############################################################

#Al ejecutar el script, hemos creado varios ficheros csv con las submissions que podemos subir 
#a drivendata y comprobar el resultado de las predicciones.

#Veamos primero cómo hemos creado los modelos para J48. Posteriormente, compararemos las
#puntuaciones F1 de ambos algoritmos en su distintas configuraciones.

#############################################################
#Creamos los modelos con cross validation con J48.

#J48 usa la máquina virtual de java y muchos recursos. En el caso de cross validation
#crearemos 10 modelos, 5 veces cada uno, es decir, ejecutaremos J48 50 veces, por lo que
#se puede demorar. 
#Todas estas tareas guardan el archivo en disco duro en caso de que se cuelgue Rstudio 
#al utilizarse tanta memoria.
#en caso de que no se quiera guardar, hay que usar la opción export = "no" (cualquiera que no sea  "yes")
#Ejecutar con precaución

#J48 con 5-fold con datos originales y todas las variables
j48.cv.wide.all = cvJ48(train, damage_grade~., "wide.all", nfolds =5)
gc()
#J48 con todas las variables datos balanced
j48.cv.wide.balanced = cvJ48(train.balanced, damage_grade~., "wide.balanced", nfolds =5)
gc()
#J48 con todas las variables con datos SMOTE
j48.cv.wide.smote = cvJ48(train.smote, damage_grade~., "wide.smote", nfolds =5)
gc()
#J48 con datos originales quitando la variable count_families
j48.cv.wide.count = cvJ48(train, damage_grade ~. -count_families, "wide.count", nfolds =5)
gc()
#J48 con datos originales quitando la variable anterior y plan_configuration
j48.cv.wide.count.plan = cvJ48(train, damage_grade ~. -count_families -plan_configuration, "wide.count.plan", nfolds =5)
gc()

###########shrunken#################
#J48 datos shrunken con todas las variables
j48.cv.long.all = cvJ48(train.long, damage_grade~., "long.all", nfolds =5)
gc()
#J48 con todas las variables datos shrunken balanced
j48.cv.long.balanced = cvJ48(train.long.balanced, damage_grade~., "long.balanced", nfolds =5)
gc()
#J48 con todas las variables con datos shrunken SMOTE
j48.cv.long.smote = cvJ48(train.long.smote, damage_grade~., "long.smote", nfolds =5)
gc()
#J48 con datos shrunken sin usar la variable count_families
j48.cv.long.count = cvJ48(train.long, damage_grade ~. -count_families, "long.count", nfolds =5)
gc()
#J48 con datos shrunken sin usar la variable anterior ni plan_configuration
j48.cv.long.count.plan = cvJ48(train.long, damage_grade ~. -count_families -plan_configuration, "long.count.plan", nfolds =5)
gc()

######################################################
#En caso de que queramos cargar los archivos o aprovechando que automáticamente se guardan en disco
archivos.j48.cv = list.files(pattern = "j48.cv.")

j48.cv.wide.all = readRDS(archivos.j48.cv[6])
j48.cv.wide.balanced = readRDS(archivos.j48.cv[7])
j48.cv.wide.smote = readRDS(archivos.j48.cv[10])
j48.cv.wide.count = readRDS(archivos.j48.cv[9])
j48.cv.wide.count.plan = readRDS(archivos.j48.cv[8])
j48.cv.long.all = readRDS(archivos.j48.cv[1])
j48.cv.long.balanced = readRDS(archivos.j48.cv[2])
j48.cv.long.smote = readRDS(archivos.j48.cv[5])
j48.cv.long.count =readRDS(archivos.j48.cv[4])
j48.cv.long.count.plan =readRDS(archivos.j48.cv[3])

#Procedemos a calcular la media de f1 de los folds

ficheros.cv.j48 = ls(pattern = "\\j48.cv*")
ficheros.cv.j48 = ficheros.cv.j48[-1]
l = list(0)
j = 0
for (i in ficheros.cv.j48){
  j= j +1
  aux = get(i)
  l[[j]] = media_f1.cv(aux)
}


cv.j48.results = data.frame(Method = ficheros.cv.j48, F1_Score = unlist(l))

#Podemos combinar los resultados de cross validation usando ambos algoritmos

#Defino los nombres bien:
names =  c("shrunken",
           "shrunken_balanced",
           "shrunken-count_families-plan_config",
           "shrunken-count_families",
           "shrunken_SMOTE",
           "raw",
           "raw_balanced",
           "raw-count_families-plan_config",
           "raw-count_families",
           "raw_SMOTE")


cv.results = data.frame(Data = names, J48 = cv.j48.results$F1_Score,  tree = cv.tree.results$F1_Score ) 


#Podemos mostrar una figura donde veamos el rendimiento comparado:

library(reshape2)
cv.results.molten =  melt(cv.results)
library(scales)
ggplot(cv.results.molten, aes( x = Data, y = value, fill = variable ) ) + 
  geom_bar( position = "identity", stat = "identity", alpha =0.7 ) + scale_fill_brewer(palette="Set2") +
  scale_y_continuous(limits=c(0.57,0.7), oob=rescale_none) + theme(axis.text.x = element_text(angle = -60, vjust = 0.5, hjust=0, size = 15)) +
  labs(x = "Data", y = "F1 score", title = "F1 score using 5-fold cross validation using training set \nfor different methods and data" ,
       fill = "Method")

#ggsave("cv.comparison.2.png", width = 5, height = 10)

###################################################################
#Creación de modelos con J48.
#Ya tenemos los modelos con cross validation para J48, ahora crearemos
#los modelos que usen todo el conjunto de datos de train para 
#crear predicciones con los valores de test y comprobarlos en la plataforma drivendata.

#Vamos a ejecutar la creación de los modelos y puede llevar bastante tiempo, recomendamos, de nuevo
#leer los archivos ya generados desde el repositorio 

#El orden es el mismo que para TREE y siguen la misma nomenclatura
j48.wide.all = j48train(train, test_x, damage_grade~., "wide.all")
j48.wide.balanced = j48train(train.balanced, test_x, damage_grade~., "wide.balanced")
j48.wide.smote = j48train(train.smote, test_x, damage_grade~., "wide.smote")
j48.wide.count = j48train(train, test_x, damage_grade ~. -count_families, "wide.count")
j48.wide.count.plan = j48train(train, test_x, damage_grade ~. -count_families -plan_configuration, "wide.count.plan")
j48.long.all = j48train(train.long, test_x.long, damage_grade~., "long.all")
j48.long.balanced = j48train(train.long.balanced, test_x.long, damage_grade~., "long.balanced")
j48.long.smote = j48train(train.long.smote, test_x.long, damage_grade~., "long.smote")
j48.long.count = j48train(train.long, test_x.long, damage_grade ~. -count_families, "long.count")
j48.long.count.plan = j48train(train.long, test_x.long, damage_grade ~. -count_families -plan_configuration, "long.count.plan")







##########################################
#Si leemos desde el archivo:

nombres.j48 = c( 
  "j48.wide.all.rds",
  "j48.wide.balanced.rds" ,
  "j48.wide.smote.rds",
  "j48.wide.count.rds" ,
  "j48.wide.count.plan.rds" , 
  "j48.long.all.rds" , 
  "j48.long.balanced.rds" , 
  "j48.long.smote.rds" , 
  "j48.long.count.rds" ,
  "j48.long.count.plan.rds")



j48.wide.all = readRDS(nombres.j48[1])
j48.wide.balanced =readRDS(nombres.j48[2])
j48.wide.smote = readRDS(nombres.j48[3])
j48.wide.count =readRDS(nombres.j48[4])
j48.wide.count.plan = readRDS(nombres.j48[5])
j48.long.all = readRDS(nombres.j48[6])
j48.long.balanced = readRDS(nombres.j48[7])
j48.long.smote = readRDS(nombres.j48[8])
j48.long.count = readRDS(nombres.j48[9])
j48.long.count.plan = readRDS(nombres.j48[10])



#En cada ejecución del archivo ya hemos producido una submission, además de que también los tenemos en el
#repositorio.
#Como nuestro principal interés es obtener una mejor clasificación, solo nos interesamos en exportar
#esta predicción y ver el score y comprobar si mejor o empeora las entregas anteriores


####################################################################
#Score F1. 
#Para cada modelo que creamos, también hemos creado un archivo submission (al usar las funciones que hemos
#enseñado anteriormente). Estos archivos las hemos ido entregando periódicamente a la plataforma
#drivendata. Mostramos aquí los dataframe de los resultados (recogidos a mano de la propia página web):



df.test.results = data.frame(J48 = c(0.7121, 0.6024, 0.6534, 0.7079, 0.7155, 0.7075, 0.5993, 0.6369, 0.7079, 0.7075),
                             tree = c(0.6383, 0.4669, 0.5955, 0.6383, 0.6383, 0.6383, 0.4632, 0.5144, 0.6383, 0.6383) )

rownames(df.test.results) = c("raw",
                              "raw_balanced",
                              "raw_SMOTE", 
                              "raw-count_families",
                              "raw-count_families-plan_config",
                              "shrunken",
                              "shrunken_balanced",
                              "shrunken_SMOTE",
                              "shrunken-count_families",
                              "shrunken-count_families-plan_config"
)



df.test.results = data.frame(Method = rownames(df.test.results), df.test.results)
rownames(df.test.results) = NULL
#Veamos los resultados de los test:
df.test.results

#Preferimos mostrarlo en un gráfico:
df.test.results.molten = melt(df.test.results)
library(scales)
ggplot(df.test.results.molten, aes( x = Method, y = value, fill = variable ) ) + 
  geom_bar( position = "identity", stat = "identity", alpha =0.7 ) + scale_fill_brewer(palette="Set2") +
  scale_y_continuous(limits=c(0.45,0.72), oob=rescale_none) + theme(axis.text.x = element_text(angle = -60, vjust = 0.5, hjust=0, size = 12)) +
  labs(x = "Data", y = "F1 score", title = "F1 score with test set from Driven Data \nfor different methods and data" ,
       fill = "Method")

#ggsave("comparison.2.png", width = 5, height = 6)


#También podemos ver los resultados de forma descendente en score obtenido:
df.test.results.molten %>% arrange(desc(value))



#Vemos que la mejor puntuación la obtiene el algoritmo J48 utilizando los datos originales
#excepto las dos variables count_families y plan_configuration.

#Vemos que tree, en general, ofrece una predicción bastante pobre. El hecho de usar los datos
#balanced, tanto en j48 como en tree hace que tengamos peor predicción, esto se debe a que
#estamos usando muchos menos datos que los originales. 
#En j48, el hecho de usar los datos originales o comprimidos no afecta demasiado, por lo 
#que podemos intuir que las variables comprimidas no son demasiado importantes para la clasificación

#SMOTE funciona mejor si no usamos los datos comprimidos, sobre todo con el algoritmo tree.
#Por otro lado, al usar los datos comprimidos y quitar las dos variables seleccionadas, no mejora la pre-
#dicción mientras que sí ocurre al usar los datos no comprimidos.


#Como vemos que la mejor puntuación la obtienen los datos originales con todas las variables menos
#count_families y plan_configuration,

#Podemos buscar para qué valor de C, es decir, el nivel de confianza en la construcción de árboles 
#del algoritmo J48, obtenemos un mejor valor de F1 en test (grid search)

#Esta búsqueda del valor óptimo de C se puede usar con librería especializadas como caret, pero hemos
#encontrado un peor rendimiento con J48 usando caret que usando directamente la función, de acuerdo a 
#resultados en internet, parece ocurrirle a más personas, por lo que esta búsqueda se ha realizado
#siguiendo la intuición (veremos que no es así).



#Definimos los valores de C para buscar: 
C_search = c(0.1, 0.15, 0.2, 0.4, 0.5, 0.6, 0.8)


#bucle para producir todos los modelos:


#Cuidado esto puede llevar bastante tiempo, ya que realizamos 5-fold cross validation en cada
#caso
for (i in C_search){
  print( i)
  model = J48(damage_grade ~. -count_families -plan_configuration, data = train, control = Weka_control(C = i) )
  cv.result = evaluate_Weka_classifier(model, numFolds = 5)
  .jcache(model$classifier)
  saveRDS(list(modelo = model, cv.resultado = cv.result), paste("grid",".C=",i, ".rds", sep=""))
}

#Podemos cargar los resultados de los archivos para producir las submissions
grid.list = list(grid.0.1 = readRDS("grid.C=0.1.rds"),
                 grid.0.15 = readRDS("grid.C=0.15.rds"),
                 grid.0.2 = readRDS("grid.C=0.2.rds"),
                 grid.0.4 = readRDS("grid.C=0.4.rds"),
                 grid.0.5 = readRDS("grid.C=0.5.rds"),
                 grid.0.6 = readRDS("grid.C=0.6.rds"),
                 grid.0.8 = readRDS("grid.C=0.8.rds")
)


for (i in c(1:length(C_search))){
  model.predict.test = predict(grid.list[[i]]$modelo, newdata=test_x)
  create_submission(model.predict.test, paste("grid.C.", C_search[i] , sep = "")) 
}

#Estos archivos podemos subirlos a la plataforma driven data y comprobar cuál es la puntuación
#que reciben.

#Los apunto a mano:

grid.results.test=data.frame(C= c(0.10, 0.15, 0.2, 0.4, 0.5, 0.6, 0.8), F1 = c(0.7205, 0.7203, 0.7184, 0.7007, 0.6950, 0.6838, 0.6838))


#Podemos graficarlo:

ggplot(data = grid.results.test, aes(x = C, y = F1)) + geom_point(stat ="identity", colour = "blue") + 
  labs( title = "Score F1 conjunto de test para J48 con distintos valores de C",
        subtitle = "damage_grade~. - count_families -plan_configuration",
        xlab = "Nivel de confianza C")

#ggsave("grid.comparison.2.png", width = 5, height = 6)

grid.results.test %>% arrange(desc(F1))

#El mejor resultado obtenido es con el nivel de confianza más bajo, lo cual nos sorprende ya que es
#demasiado bajo, obteniendo una clasificación score final de 0.7205 en DrivenData. 

#Con esto finaliza la parte de árboles de decisión para este problema, recalcar que con técnicas como
#randomForest se consiguieron valores de 0.7305, mejorando esta clasificación, aunque, como no se pueden
#usar en la memoria, no se incluyeron.

#Por otra parte, en este script aparece la generación de los modelos de una forma ``sistemática", durante
#el proceso de la práctica se fueron probando los modelos de una forma mucho menos directa, conforme se iban ocurriendo,
#aunque lo mostremos así en este archivo.

#En la cuenta de drivendata original, se hicieron un total de 28 subidas, las que se tienen aquí mas la de
#random forest de prueba.

#Como ya comenté por mail, intentaré subir hasta el día de la presentación las 27 restantes( como no da tiempo)
#subiré las 20 originales más 4 de grid, en una nueva cuenta para que se vea la evolución y la clasificación 
#final 

#repositorio: https://github.com/Rumoa/preproc_classif/
#cuenta original: https://www.drivendata.org/users/toni
#nueva cuenta: https://www.drivendata.org/users/toni_ruiz/