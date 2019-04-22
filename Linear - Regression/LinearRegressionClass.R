temp = c(24,26,31,27,29,31,30,36,33,29,27,34,26)
city = c(rep('A',4),rep('B',5),rep('C',4))

mean(temp)

df = data.frame(temp,city)
aggregate(temp~city,data=df, FUN = mean)

a = aov(temp ~ city, data = df)  
summary(a)

names(a)
a$coefficients


ins = read.csv(file.choose())
View(ins)

head(ins)

cor(ins$age,ins$charges)
aggregate(charges ~ sex, data = ins,FUN = mean)

library(ggplot2)
ggplot(ins,aes(sex,charges)) + geom_boxplot()

cor(ins$age,ins$bmi)

hist(ins$charges)
hist(log(ins$charges))
hist(sqrt(ins$charges))

ins$charges = log(ins$charges)

set.seed(234)
id= sample(1:10, 5)
id

?sample
ids = sample(nrow(ins),nrow(ins)*0.8)
ids
?sample_n
df = sample_n(ins,size = 1070)
View(df)
OR
df1 = sample_frac(ins,0.8)
View(df1)


train = ins[ids,]
View(train)
test = ins[-ids,]
View(test)

#Multicolliarity: if input variables are dependant 
# caluculate correlation of variables
# If we have 100's of varibales then we cal VIF
model1 = lm(charges ~ ., data = train) 

# If variable has p value greater than 0.05 then ignore that variable and bui model again 
# Other way declaration(charges ~ age + bmi, data = train)
#other way Declartion(charges ~ .-bmi,data = train)
summary(model1)

test$pred = predict(model1,newdata= test)

View(test)
test$err = test$charges - test$pred

#mean square error
mean(test$err**2)

# RMSE
sqrt(mean(test$err**2))

##mape mean absolute percentage error
#abs - absolute
# Anything below 5% error its OK to consider
mean((abs(test$err)/test$charges)*100)

## model dianostics

##inear realtionship
plot(ins$age, ins$charges)
cor(ins$age,ins$charges)

## Multi colliearity
names(ins)
?vif
vif(lm(bmi ~ age +children, data=train))

l## multicollinearirity is not a maor prob
## constant variance of errors terms and auro corrrelation

std.res = stdres(model1)
pred = model1$fitted.values


## outliers test
outlierTest(model1)

## high leverage obser.
plot(model1,which = 4)

## rebuilt model
## age is not having a linear reatinshp with charges
ins$age_s = sqrt(ins$age)
plot(ins$age_s,ins$charges)
hist(ins$age)

## Convert age to a ordinal variable
ins$agegrp = ifelse(ins$age <= 20,'Le20',ifelse(ins$age<=40,'20to40',ifelse(ins$age<=60,'40to60','60+')))
table(ins$agegrp)


missed

##normality of errors
##qqplot

?plot
boxplot(ins$charges)
plot(model2,which =2)

############ Linear Regression practice Class##############
install.packages('Ecdat')
library(Ecdat)
data(Computers)
head(Computers)
nrow(Computers)
dim(Computers)
com = Computers
dim(com)
View(com)
summary(com)
str(com)

## target variable should be Y axis
## Price is the target variable
#ED

hist(com$price)
hist(log(com$price))
hist(sqrt(com$price))
com$price = log(com$price)

## Multiple variables correlation
names(com)
cor(com[,c(2,3,4,5,9,10)])

cor(com$price, com$speed)
cor(com$price,com$hd)
cor(com$price,com$ram)
# cor(com$price,c(2,3,4,5,6))
cor(com$speed,com$hd)
cor(com$ram,com$screen)

# cor(com$price[,c(numeric values)])

boxplot(price ~cd,data = com)
ggplot(com,aes(cd,price)) + geom_boxplot()
ggplot(com,aes(multi,price))+ geom_boxplot()
ggplot(com,aes(premium,price)) + geom_boxplot()

com$screen = as.factor(com$screen)
ggplot(com,aes(screen,price)) + geom_boxplot()

cor(com$price,com$ram)

summary(com$ads)

cor(com$price,com$ads)

summary(com$trend)
cor(com$price,com$trend)

##linear relationship
## Linear relationship is straight line 
## If it is curve shape then it is non linear
plot(com$speed,com$price)
ggplot(com,aes(speed,price)) + geom_point()

plot(com$hd,com$price)

###divide the data

set.seed(123)
ids = sample(nrow(com),nrow(com)*0.8)
train = com[ids,] ## 80%
View(train)
summary(train)

test = com[-ids,]  ## remaining 20%
View(test)

m1 = lm(price ~ .,data= train)

summary(m1)

m2 = lm(price ~ .-ads-trend-multi,data = train)
summary(m2)

###Diagnostics
cor(train[,c(2,3,4,5)])
cor(train$hd,train$ram)
## Variance inflation factor
vif(lm(hd ~ screen + ram, data = train))

?stdres ## Extract standardized residuals from LM
std.err = stdres(m2)
?plot
plot(m2,which = 2)
hist(std.err)

## Heteroscdasticity
names(m2)
pred = m2$fitted.values
plot(pred,std.err)

plot(m2,which = 1)

plot(m2, which = 3)

durbinWatsonTest(m2)

outlierTest(m2)

plot(m2, which = 4)

## Test dataset

test$pred = predict(m2,newdata = test)
# test$pred = predict(model1,newdata= test)

View(test)
test$err = test$price - test$pred

#mean square error
mean(test$err**2)

# RMSE
sqrt(mean(test$err**2))

#MAPE
mean((abs(test$err)/test$price)*100)
