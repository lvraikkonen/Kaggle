train <- read.csv("~/Kaggle Totorial/train.csv", stringsAsFactors=FALSE)
test <- read.csv("~/Kaggle Totorial/test.csv", stringsAsFactors=FALSE)

str(train)
table(train$Survived)

## proportion
prop.table(table(train$Survived))

## add surived column to test dataset(418 test items)
test$Survived <- rep(0,418)

table(train$Sex)
prop.table(table(train$Sex, train$Survived))
## proportions in the 1st dimension which stands for the rows 
prop.table(table(train$Sex, train$Survived),1)

test$Survived[test$Sex == 'female'] <- 1

##############################################
train$Child <- 0
train$Child[train$Age < 18] <- 1

aggregate(Survived ~ Child + Sex, data=train, FUN=sum)
aggregate(Survived ~ Child + Sex, data=train, FUN=length)
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '<10'

aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] <- 0

######################################################
library(rpart)

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")
#######
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train,
             method="class", control=rpart.control(minsplit=2, cp=0))
new.fit <- prp(fit,snip=TRUE)$obj
fancyRpartPlot(new.fit)

predicition <- predict(fit, test, type="class")

######################################################
## naive Bayes
library(e1071)

model <- naiveBayes(Survived ~ Sex + Pclass + Age + Fare, data = train)
pred <- predict(model, test)

######################################################
## random forest
library(randomForest)

fit <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                    data=train, importance=FALSE, ntree=2000)


submit <- data.frame(PassengerId = test$PassengerId, Survived = pred)
write.csv(submit, file = "pred.csv", row.names = FALSE)
