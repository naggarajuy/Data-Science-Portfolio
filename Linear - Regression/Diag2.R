
#### dataset avilable in CAR Package
library(ISLR)
library(car)
library(MASS)
head(mtcars)
dim(mtcars)

### select only a few variable 

fit = lm(mpg  ~  hp+wt , data=mtcars)
summary(fit)
### Assumtions of linear regression 
## Input and target variables should have linear relationship 
## 1. Normal distribution of target variable : Yes 
## 2. Input variables independent from each other : Fail
## 3. Constant variance of error terms(No Heteroscedasticity) : Fail
## 4. Normality of errors : Fail 
## 5. Autocorrelation : Fail 
## 6. Outlier test and Leverages : 


## Multicollenairity check using VIF 

vif(lm(hp ~ wt+cyl, data=mtcars)) ## VIF is less than 5 , so ignorable 
vif(lm(disp ~ cyl + drat + wt + hp, data=mtcars))
## we cant use both Cyl and Disp as input variables 
## apply feature engineering to bring these varaibles to one variables 

## check for normality of errors 
names(fit)
errors = fit$residuals
## histogram of errors 
hist(errors)
## qq plot on errors 
qqPlot(fit, main="QQ Plot")

## Constant variance pf error terms and Autocorrelation 

## a plot of std.residuals and predicted values 
predicted = fit$fitted.values ## predicted values on training data 
std.res = stdres(fit)

plot(predicted , std.res)

plot(fit, which = 1) ## a family of plots for diagnosis 
plot(fit, which = 2)
plot(fit, which = 3)
plot(fit, which = 4)

### Outlier detection test 

outlierTest(fit)

## for autocorrelation we have durbin-watson test 

durbinWatsonTest(fit)
