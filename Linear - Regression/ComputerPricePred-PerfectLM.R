## Check NA's
## Outliers treatment
## Check normalization of target variable
## Check correlation b/w input variables
## linear relationshp 
install.packages('Ecdat')
library(Ecdat)
data(Computers)
View(Computers)
head(Computers)
dim(Computers)

# To get missing values count column wise
colSums(is.na(Computers))
sum(is.na(Computers))
sum(is.na(Computers$price))
summary(Computers)

hist(Computers$price)

## linear regression model to predict the price of a computer 

## check for normal distrubution of price variable 

library(ggplot2)

ggplot(Computers, aes(price)) + geom_histogram(bins = 30)

#Check target variable is normalised
hist(Computers$price)

hist(log(Computers$price))
hist(sqrt(Computers$price))

##Outliers
boxplot(Computers$hd)$out
x=Computers$hd
qnt <- quantile(x, probs=c(.01,.02,.03,.04,.05,.1,.15,.20,.25,
                           .30,.35,.40,.45,.50,
                           .55,.60,.65,.70,.75,
                           .80,.85,.90,.95,.98,.99,1),decreasing =T, na.rm = T)
qnt
caps <- quantile(x, probs=c(.05, .95), na.rm = T)
caps

H <- 1.5 * IQR(x, na.rm = T)
H

x[x < (qnt[9] - H)] =caps[1] # For below values
x[x > (qnt[19] + H)] = caps[2] # For above values
caps[2]
boxplot(x)
Computers$hd = x
### Input variables not correlated with each other 

names(Computers)

#Check cor b/w all variables
cor(Computers[,c(2,3,4,5,9,10)])

## ram and hd are having strong correlation
## Look at vif(variance inflation factor) , Vif is less than 2 so ignorable 
install.packages('car')
library(car)
?vif
vif(lm(ram~hd + speed, data=Computers))

vif(lm(hd~ram + trend, data=Computers))

#### 
summary(Computers$hd)
summary(Computers$ram)

var(Computers$hd)
var(Computers$ram)

## only use hd in the model 

### linear relationship 

plot(Computers$price, Computers$speed)

plot(Computers$price, Computers$hd)

plot(Computers$ram, Computers$price)


## check for NA 
sum(is.na(Computers))

### Model 

set.seed(1245)

ids = sample(nrow(Computers), nrow(Computers)*0.8 )

train = Computers[ ids, ] ## 80% of data into train
test = Computers[ -ids,] ## 20% of data into test

### Linear model 

pricemodel = lm( price ~ . , data=train)
## Take the input varibles whose p value is < = 0.05
## If one variable is satisfying that condition remove that variable and do modelling
 ## with other variables.Repeat the same process till variables satisfying condition 
## Now do modelling with all variables p value less than 0.05
## Check R square value should be near to 1
summary(pricemodel)

### Performance on the test dataset 

test$pred = predict(pricemodel, newdata = test )

### metrics RMSE, and MAPE 

test$error = test$price - test$pred

mse = mean(test$error **2)

rmse = sqrt(mse)
(or)
rmse = sqrt(mean(test$error ** 2))

## RMSE value should be near to '0' then we can say model is working fine
rmse

## mape (mean absolute percentage error) value should be near to '0'

test$abserr = abs(test$error)

test$pererr = (test$abserr/test$price) * 100

mape = mean(test$pererr)

mape


## Model diagnostices 

library(MASS)

names(pricemodel)

## normality of errors 

hist(pricemodel$residuals)

### PLot QQ plot 

plot(pricemodel, which=2)

### checking auto correlation and heteroscedasticity 

res = stdres(pricemodel)

pred = pricemodel$fitted.values

plot( pred, res)

#### Normality of errors 

hist(res)


### plot from models 

plot(pricemodel, which = 1)

### outliers test 

outlierTest(pricemodel)

### Target variable to nomral distribution 

hist(log(Computers$price))


Computers$logprice = log(Computers$price)

price = Computers$price

### delete price variable from df 

Computers$price = NULL

## divide data into train and test

set.seed(1245)

ids = sample(nrow(Computers), nrow(Computers)*0.8 )

train = Computers[ ids, ]
test = Computers[ -ids,]

### model 

logmodel = lm( logprice ~ . , data=train)

summary(logmodel)

### model diagnostics 

hist(logmodel$residuals)

plot(logmodel, which = 2)


logres = stdres(logmodel)
logpred = logmodel$fitted.values

plot(logpred, logres)

plot(logmodel, which = 1)

### outliertest 

outlierTest(logmodel)

### higleverage observations 

cd = cooks.distance(logmodel)
cd
library(MASS)
library(ISLR)
library(car)
### outliertest 
outlierTest(logmodel)

length(logmodel$coefficients)
cutoff = 4/(nrow(train) - length(logmodel$coefficients) )
cutoff = 4/(5007 - 10)
cutoff
summary(logmodel)

plot(logmodel, which = 4, cutoff = '0.0008')

### perfomance of logmodel 

test$logpred = predict(logmodel, newdata = test)

test$logerr = test$logprice - test$logpred

sqrt(mean(test$logerr**2))

### mape 

### mape on log scale 

test$abslogerr = abs(test$logerr)

test$logper = (test$abslogerr/test$logprice)*100

mean(test$logper)


## convert the logprice and predictions into actual values 
?exp
test$price = exp(test$logprice)

test$pred = exp(test$logpred)

### metrics RMSE, and MAPE 

test$error = test$price - test$pred

rmse = sqrt(mean(test$error ** 2))

rmse

## mape 

test$abserr = abs(test$error)

test$pererr = (test$abserr/test$price) * 100

mape = mean(test$pererr)

mape

#### improvise the model  

logmodel1 = lm(logprice ~ . -cd -multi -ads , data = train)

summary(logmodel1)

names(train)

### remove outlier and influencers from training data 

train2 = train[-c(4478,3784,5961), ]


summary(train)
summary(logmodel1)

### rebuild the model 

newmodel = lm(logprice ~ . -ram, data=train2)

summary(logmodel)
summary(newmodel)

cd = 4/(nrow(train2) - 9)
logres = stdres(newmodel)

logpred = logmodel$fitted.values

outlierTest(logmodel)

plot( logpred, logres)

plot(logmodel, which = 4)

###  Multicollnearity 

vif( lm(hd ~ ram + screen, data=train2))

model2  = lm( logprice ~ . , data=train2)

summary(model2)

#### predictions using model2 

test$pred = predict( model2, newdata = test)

test$err = test$logprice - test$pred

test$errsq = test$err ** 2

sqrt(mean(test$errsq))

mean(test$logprice)

0.11/7.6

