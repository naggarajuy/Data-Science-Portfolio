
#### dataset avilable in CAR Package

library(ISLR)
??car
install.packages('car')
library(car)
head(mtcars)
head(mtcars)
dim(mtcars)

hist(mtcars$mpg)

### Target variable - Normal distribution 

hist(mtcars$mpg)


## inverse transformation
hist(1/(mtcars$mpg))


### relationship b/w input and target variabes 

plot(mtcars$hp, mtcars$mpg)

cor(mtcars)

#Input variables are dependant is caled multicollinearity
### Mullticollinearity 

cor( mtcars[ , -1 ])

head(mtcars)
hist(mtcars$mpg)

cor(mtcars$carb, mtcars$am)

### select only a few variable 

fit = lm(mpg  ~  hp+wt , data=mtcars)


#### correlation check or Multicollinearity 
summary(fit)
 cor(mtcars)

### check for auto correlation and heteroscedasticity

names(fit)
 
fit$coefficients

fit$residuals

hist(fit$residuals)

names(fit)

fit$coefficients

hist(fit$residuals) ### normality of errors

library(MASS) ## need to calculate standardised residuals

residuals = stdres(fit) ## Studentised residuals 
summary(residuals)

hist(residuals)

mtcars$mpg_s = scale(mtcars$mpg)

summary(mtcars$mpg_s)


### predicted values vs. fitted values 

plot(fit$fitted.values, residuals )

durbinWatsonTest(fit)  ## to check auto correlation

?durbinWatsonTest
?outlierTest
outlierTest(fit)

### MAAS 

mtcars$normmpg = scale(mtcars$mpg)

summary(mtcars$normmpg)
## Model summary
summary(fit)

### normal distirbution check
hist(log(mtcars$hp))

mtcars

### To check it is following normnal distri # percentage of points on the line then we can udge

qqPlot(fit, main="QQ Plot") #qq plot for studentized resid 
leveragePlots(fit) # leverage plots


#### leverage statistics

cd = cooks.distance(fit)

cd

mtcars
head(cd)
### Cooks distance for high leverage points
#cutoff <- 4/((nrow(mtcars)-length(fit$coefficients)-2))

cutoff <- 4/((nrow(mtcars)-2 -1)) 

cd > cutoff
cutoff
?cooks.distance
##
plot(fit, which=6, cook.levels=cutoff) ## Which takes ( 1, 2, 3, 4, 5, 6)

head(mtcars)


## Outlier detection
outlierTest(fit)


### Variance inflation factor to check multicollinearity 
vif(fit)


#### identify the observations that are outliers and leverages

mtcars$rownum =seq(1:nrow(mtcars))
head(mtcars)

mtcars

mtcars = mtcars[-c(17,20,31),]

dim(mtcars)

mtcars
names(fit)
summary(mtcars)
mtcars[17,]

## Non -normality
sresid <- studres(fit)
stres = stdres(fit)

## Residual plots

plot(fit$fitted.values, fit$residuals )

xfit<-seq(min(sresid),max(sresid),length=40) 
yfit<-dnorm(xfit) 
lines(xfit, yfit)
names(fit)


fit$residuals


### tests for diagnosis

##Multicollienearity
vif(fit)
sqrt(vif(fit)) > 2

cor(mtcars$disp, mtcars$wt)
##Non-independence of errors

durbinWatsonTest(fit) ## Autocorrelation
?durbinWatsonTest
  ## Outlier test 

outlierTest(fit)
