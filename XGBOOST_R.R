
library(mlbench)
library(caret)

data("Sonar")
df = Sonar

help(Sonar)
dim(df)
str(df)
summary(df)

##Sampling
set.seed(0)
ids = sample(nrow(df),nrow(df)*0.75)
train = df[ids,]
test = df[-ids,]

##Cross Validaion parameters
ControlParameters = trainControl(method = 'cv',number = 5, savePredictions = T,
                                 classProbs = T)

##Model parameters
# ParametersGrid = expand.grid(eta=0.1,colsample_bytree =c(0.5,0.7), max_depth= c(3,6),
                             # nrounds= 100, gamma=1, min_child_weight=2, subsample = 0.8)
ParametersGrid = expand.grid(eta=0.1,colsample_bytree = 0.7, max_depth= 4,
                              nrounds= 150, gamma=1, min_child_weight=2, subsample = 0.8)

##Model building
modelXGBoost = train(Class ~ ., data=train, method='xgbTree', trControl= ControlParameters,
                     tuneGrid = ParametersGrid)

modelXGBoost

##Predictions
pred = predict(modelXGBoost, test)
table(actual = test$Class,predictions= pred)
