##### Loan prediction #####

#Loading CARET package
library(caret)

# Loading training data
train_F = read.csv(file.choose(),stringsAsFactors = T)

str(train_F)

#Check for missing values
colSums(is.na(train_F))
sum(is.na(train_F))
table(is.na(train_F$Credit_History))

prop.table(table(train_F$Credit_History))

#Imputing missing values using KNN.Also centering and scaling numerical columns
preProcValues <- preProcess(train_F, method = c("knnImpute","center","scale"))
?preProcess
install.packages("RANN")
library(RANN)
train_processed = predict(preProcValues,train_F)
sum(is.na(train_processed))

train_processed$Loan_Status = ifelse(train_processed$Loan_Status == 'N',0,1)

train_processed$Loan_ID = NULL
str(train_processed)

##Convering every categorical variable to numerical 
dmy = dummyVars('~.',data = train_processed, fullRank = T)
train_transformed = data.frame(predict(dmy,newdata = train_processed))

str(train_transformed)

train_transformed$Loan_Status = as.factor(train_transformed$Loan_Status)

###Spliting data
ids = sample(nrow(train_transformed), nrow(train_transformed)*0.80)
train = train_transformed[ids,]
test = train_transformed[-ids,]

##Feature selection using RFE(recursive feature elimination) in R
control = rfeControl(functions = rfFuncs, method = 'repeatedcv',repeats = 3,verbose = F)
outcome = 'Loan_Status'

?rfe
predictors = names(train)[!names(train) %in% outcome]
loan_pred_profile = rfe(train[,predictors], train[,outcome], rfeControl = control)
loan_pred_profile

predictors = c('Credit_History','ApplicantIncome','CoapplicantIncome','LoanAmount','Property_Area.Semiurban')

##Model building
model_gbm = train(train[,predictors],train[,outcome],method ='gbm')
model_rf = train(train[,predictors],train[,outcome],method = 'rf')
model_nnet = train(train[,predictors],train[,outcome],method = 'nnet')
model_glm = train(train[,predictors],train[,outcome],method = 'glm')

## Variable imp estimation using CARET
# varImp(model_gbm)
varImp(object=model_rf)
plot(varImp(model_rf), main= 'RF - variable importance')

varImp(model_glm)
plot(varImp(model_glm),main = 'GLM - variable importance')

varImp(model_nnet)
plot(varImp(model_nnet),main = 'NNET - variable importance')

##Predections
pred_gbm = predict(model_gbm,newdata = test)
table(Actual=test$Loan_Status, pred=pred_gbm)

pred_rf = predict(model_rf, newdata = test)
table(actual =test$Loan_Status,pred=pred_rf)

pred_nnet = predict(model_nnet, newdata = test)
table(actual =test$Loan_Status,pred=pred_nnet)

pred_glm = predict(model_glm, newdata = test)
table(actual =test$Loan_Status,pred=pred_glm)

###Using train_f train data we will test results of test data(actual)
test_f = read.csv(file.choose(),stringsAsFactors = T)

str(test_f)

#Imputing missing values using KNN.Also centering and scaling numerical columns
preProcValues_test <- preProcess(test_f, method = c("knnImpute","center","scale"))

test_f_processed = predict(preProcValues_test,test_f)
sum(is.na(test_f_processed))

test_f_processed$Loan_ID=NULL
##Convering every categorical variable to numerical 
dmy = dummyVars('~.',data = test_f_processed, fullRank = T)
test_f_transformed = data.frame(predict(dmy,newdata = test_f_processed))

str(test_f_transformed)

test_f$Loan_Status = predict(model_rf, newdata = test_f_transformed)
table(test_f$Loan_Status)

write.table(test_f,'test_f.csv',sep = ",", row.names = F)
