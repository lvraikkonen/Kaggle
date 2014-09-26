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

pred.decissiontree <- predict(fit, test, type="class")

######################################################
## naive Bayes
library(e1071)

naivebayes.fit <- naiveBayes(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train)

pred.naivebayes <- predict(naivebayes.fit,test)

## Submit result
submit <- data.frame(PassengerId = test$PassengerId, Survived = pred.naivebayes)
write.csv(submit, file = "pred.csv", row.names = FALSE)
######################################################
## random forest
library(randomForest)

fit <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                    data=train, importance=FALSE, ntree=2000)


submit <- data.frame(PassengerId = test$PassengerId, Survived = pred)
write.csv(submit, file = "pred.csv", row.names = FALSE)



############################remove NAs###########################
mean.train.age <- mean(train$Age, na.rm=TRUE)
train$Age[is.na(train$Age)] <- mean.train.age

mean.test.age <- mean(test$Age, na.rm=TRUE)
test$Age[is.na(test$Age)] <- mean.test.age

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")
pred.decissiontree <- predict(fit, test, type="class")

submit <- data.frame(PassengerId = test$PassengerId, Survived = pred.decissiontree)
write.csv(submit, file = "pred_decisiontree.csv", row.names = FALSE)

################## Feature Engineering ##########################
head(train$Name)

test$Survived <- NA
conbi <- rbind(train,test)

strsplit(conbi$Name[1],split='[,.]')[[1]][2]
conbi$Title <- sapply(conbi$Name, FUN= function(x){strsplit(x, split='[,.]')[[1]][2]})
conbi$Title <- sub(' ','',conbi$Title) ## trim
conbi$Title[conbi$Title %in% c('Mme','Mlle')] <- 'Mlle'
conbi$Title[conbi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
conbi$Title[conbi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

conbi$Title <- as.factor(conbi$Title)

## add the number of siblings, spouses, parents and children the passenger 
conbi$FamliySize <- conbi$SibSp + conbi$Parch + 1
conbi$Surname <- sapply(conbi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})

conbi$FamilyID <- paste(as.character(conbi$FamliySize), conbi$Surname, sep="")
conbi$FamilyID[conbi$FamliySize <= 2] <- 'Small'
table(conbi$FamilyID)


famIDs <- data.frame(table(conbi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]

conbi$FamilyID[conbi$FamilyID %in% famIDs$Var1] <- 'Small'
conbi$FamilyID <- factor(conbi$FamilyID)


## break them apart 
train <- conbi[1:891,]
test <- conbi[892:1309,]

## Prediction
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamliySize + FamilyID,
             data=train, method="class")
pred.decissiontree <- predict(fit, train, type="class")

## submit
submit <- data.frame(PassengerId = test$PassengerId, Survived = pred.decissiontree)
write.csv(submit, file = "pred_decisiontree.csv", row.names = FALSE)