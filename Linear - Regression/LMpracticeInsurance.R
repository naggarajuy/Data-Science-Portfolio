ins = read.csv(file.choose())
dim(ins)
str(ins)
table(ins$children)

#Check for NA's
summary(ins)

# Target varaible shoul be Normal distribution
boxplot(log(ins$charges))
hist(ins$charges)
hist(log(ins$charges))

## Creating new variable with log value
ins$log_charges = log(ins$charges)

plot(ins$age,ins$log_charges)
plot(ins$age,ins$charges)
plot(ins$region,ins$log_charges)

library(corrplot)
corrplot(ins, method='circle')
cor(ins$age,ins$log_charges)
cor(ins$bmi,ins$log_charges)
str(ins)

ins = ins[,-7]

##Sampling
ids = sample(nrow(ins),nrow(ins)*0.8)
train = ins[ids,]
test = ins[-ids,]

##modelling
model = lm(log_charges ~ .-sex,data = train)
summary(model)

model$fitted.valu
