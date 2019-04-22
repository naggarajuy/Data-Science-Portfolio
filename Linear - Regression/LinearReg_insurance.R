setwd("D:/AP/linear")

insurance = read.csv("insurance.csv")

insurance = read.csv(file.choose(),stringsAsFactors = F)  
head(insurance)
View(insurance)
dim(insurance)
### pre checks 

sum(is.na(insurance))
### target variable should follow normal distribution 

hist(insurance$charges)

### target variable is right skewed 
## apply transformation 

hist(log(insurance$charges))
insurance$log_charges = log(insurance$charges)

hist(insurance$log_charges)


#hist(sqrt(insurance$charges))
### linear relationship b/w input and target 

plot(insurance$age, insurance$log_charges)
plot(insurance$age , insurance$charges)

### correlation b/w input and target variable

cor( insurance$age, insurance$log_charges) ## Correlation with target variable 


### multicollinearity ( input variables are correlated with each other)

cor( insurance$age, insurance$bmi) ### cor with input variable ( not desirable)
 
library(ggplot2)
 
ggplot( insurance, aes( smoker, log_charges)) + geom_boxplot()


cor(insurance$age, insurance$bmi)
### MOdel bulding  

insurance$charges = NULL 
insurance=insurance[,-7]

## train and test set split 

set.seed(675)

ids = sample( nrow(insurance), nrow(insurance)*0.8)

train = insurance[ids,]
test = insurance[-ids,]
View(test)
## model 

lin_model = lm(log_charges ~ . , data=train )

summary(lin_model)

## Test the model 
test$pred = predict(lin_model, newdata=test )

### RMSE value
test$error = test$log_charges - test$pred

test$error_sq = test$error ** 2

rmse = sqrt(mean(test$error_sq))
rmse

# R square value check if correct
sqrt(cor(test$log_charges,test$pred))

summary(test$log_charges)
0.43/9.13

#####Assumptions of linear regression
# 1. Normal distribution of target variable: Yes
# 2. Input variables are independant from each other fail
# 3. Constant variance of error terms(No heteroscedasticity)
# 4. normality of errors: Fail
# 5. Auto correlation : fail
# 6. Outlier test and leverages :

###### Diagnosis #####################

### select only a few variable 

fit = lm(log_charges  ~  . , data=train)


#### correlation check or Multicollinearity 
summary(fit)
names(fit)

fit$coefficients

head(fit$residuals)

### check for normality of errors 
hist(fit$residuals)

### check for auto correlation and heteroscedasticity

library(MASS) ## need to calculate standardised residuals

residuals = stdres(fit) ## Stadentised residuals 
summary(residuals)

### predicted values vs. fitted values 
plot(fit$fitted.values, residuals )

### statistical test for autocorelation
durbinWatsonTest(fit)  ## to check auto correlation

##### Outliers test
outlierTest(fit)

### leverage statistics ( cooks.distance)
cd = cooks.distance(fit)
 
cutoff = 4/( nrow(train) - length(fit$coefficients) )
 
### plot for finding the obs which has high leverage(using cd) 
plot(fit, which=4, cook.levels=cutoff)

#Outlier and levergae observations
 
 431, 220, 1028, 1040, 103,527, 1020, 
### Variance inflation factor to check multicollinearity 
vif(fit)

 
 ## rebuild the model by removing outlier observations 
 
 train = train[ -c( 431, 220, 1028, 1040, 103,527, 1020), ]
 
 
 ## fit the model 
 
fit = lm(log_charges  ~  . -bmi , data=train)
 
summary(fit)
 
## check if Autcorrelation or Heteroscedasticty has improved 
plot(fit, which=3)
 
##  explore the relationship b/w target and input 
 
ggplot( train, aes( bmi, log_charges)) + geom_point()
 
ggplot( train, aes( age, log_charges)) + geom_point()
 
### combine bmi and age into a new variable 
 
train$age_bmi = sqrt(train$bmi/train$age)

ggplot( train, aes( sqrt(age_bmi), log_charges)) + geom_point()
 
#### final model with ratio of age and bmi 
 
fit = lm( log_charges ~ . -bmi -age,  data=train)
summary(fit) 

train$bmi.age = log( train$bmi*train$age)

ggplot( train, aes( sqrt(bmi.age), log_charges)) + geom_point()


fit = lm( log_charges ~ . -bmi -age -age_bmi, data=train)

summary(fit)

plot(fit, which = 1)

### 

?durbin.watson
