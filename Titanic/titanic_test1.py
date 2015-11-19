import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

# Print you can execute arbitrary python code
train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )


def create_submission(alg, train, test, predictors, filename):

    alg.fit(train[predictors], train["Survived"])
    predictions = alg.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv(filename, index=False)


def process_data(titanic):

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Age"].median()

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    # Generating a familysize column
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

    # The .apply method generates a new series
    titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
    titanic["Age2"] = titanic["Age"].apply(lambda x: x * x)

    return titanic

train_data = process_data(train)
test_data = process_data(test)


predictors = ["Pclass", "Sex", "Age", "SibSp",
              "Fare", "FamilySize", "NameLength", "Age2"]
forest = RandomForestClassifier(
    n_estimators=512, max_depth=None, min_samples_split=8, random_state=0)
forest.fit(train_data[predictors], train_data["Survived"])
scores = cross_validation.cross_val_score(
    forest,
    train_data[predictors],
    train_data["Survived"],
    cv=3
)

print(scores.mean())

create_submission(forest, train_data, test_data,
                  predictors, "titanic_test1.csv")
