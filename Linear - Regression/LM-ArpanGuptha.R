## Arpan guptha Regression in R

library(ggplot2)
data(mtcars)
str(mtcars)

df = mtcars[,c(1,3,4,6)]
plot(df)
plot(df$mpg ~ disp)

model = lm(mpg ~ disp, data = df)
summary(model)
abline(model)

model = lm(mpg ~ disp+hp+wt, data = df)
summary(model)
abline(model)

model = lm(mpg ~ hp+wt, data = df)
summary(model)
abline(model)