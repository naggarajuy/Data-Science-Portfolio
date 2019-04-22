## Anova and linear regression
temp = c(24,26,31,27, 29,31,30,36,33,29,27,34,26)
city = c(rep("A",4), rep("B",5),rep("C",4))

df = data.frame(temp, city)

mean(temp)

aggregate(temp ~ city, data = df, FUN=mean)

## ananlysis of variance 

a = aov(temp ~ city, data = df)
summary(a)


## linear regression 
setwd("D:/AP/linear")

ins = read.csv("insurance.csv")
ins = read.csv(file.choose())
head(ins)
## EDA 
cor(ins$age, ins$charges)

## between gender and charges 
aggregate(charges ~ sex , data = ins, FUN = mean)

## Box plot 

library(ggplot2)

ggplot(ins, aes(sex, charges)) + geom_boxplot()

## input variables are independent from each other 

cor(ins$age, ins$bmi)

### Target variable normal distribution 

hist(ins$charges)

hist(log(ins$charges))
hist(sqrt(ins$charges))

ins$charges = log(ins$charges)
##  split the data into train and test sets 

set.seed(234)

ids = sample(nrow(ins), nrow(ins)*0.8)
train = ins[ids,]
test = ins[-ids,]


### model 
model1 = lm( charges ~ . , data = train)
summary(model1)
## model testing 
test$pred = predict(model1, newdata = test)
## model performance 
test$err = test$charges - test$pred
mean(test$err**2)
sqrt(mean(test$err**2))
## mape 
mean((abs(test$err)/test$charges)*100)

## model diagnostics 

## linear relationship 
plot(ins$age , ins$charges )
cor(ins$age, ins$charges)

### multicollinearity 
names(ins)
vif(lm(bmi ~ age + children, data = train))
## multicollinearity is not a major problem 
### Constant variance of error terms and autocorrelation 

hist(model$residuals)
std.res = stdres(model1)
pred = model1$fitted.values

plot(pred, std.res)

### Normality of errors 
plot(model1, which = 2) ## qqplot 

## outlier test 

outlierTest(model1)

## high leverage observations 

plot(model1, which = 4)


### rebuilt the model 
## age is not having a liner relationship with charges

ins$age_s = sqrt(ins$age)
plot(ins$age_s, ins$charges)
hist(ins$age)

## convert age to a ordinal variable 

ins$agegrp = ifelse(ins$age <=20, "Le20", ifelse(ins$age<=40,"20to40",ifelse(ins$age <=60,"40to60","60Plus")))

table(ins$agegrp)

## check b/w bmi and charges 

plot(ins$bmi, ins$charges)

## drop age and age_s variables
ins$age_s = NULL
ins$age = NULL
## create train and test datasets 

set.seed(234)

ids = sample(nrow(ins), nrow(ins)*0.8)
train = ins[ids,]
test = ins[-ids,]

## model2 

model2 = lm(charges ~ . , data = train)
summary(model2)

### Model diagnostics 

## normality of errors 
## qq plot 

plot(model2, which = 2)

plot(model2, which = 1)

## outlier 

outlierTest(model2)

plot(model2, which = 4)
## remove the observations which are outlier
## 517, 431, 1040, 103

### eliminate these observation in the model 

train[1040, ]
train[103,]
train[517,]
train[431,]

hist(train$charges)

train_ex = train[train$agegrp == "Le20" & train$charges > 10.1, ]

train = train[!(train$agegrp == "Le20" & train$charges > 10.0), ]


### Model 

model3 = lm(charges ~ . , data = train)
summary(model3)

## Error diagnostics 

plot(model3, which = 2)

plot(model3, which = 4)

### constant variance of error terms and Autocorelation 

plot(model3, which = 1)
plot(model3, which = 3)

