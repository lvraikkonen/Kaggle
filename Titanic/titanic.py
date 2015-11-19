import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Clean Data and Feature Selection


def clean_data(df, drop_passenger_id):

    # Get the unique values of Sex
    sexes = df['Sex'].unique()

    # Generate a mapping of Sex from a string to a number representation
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))

    # Transform Sex from a string to a number representation
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)

    # Transform Embarked from a string to dummy variables
    df = pd.concat(
        [df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val')], axis=1)

    # Fill in missing values of Embarked
    # Since the vast majority of passengers embarked in 'S': 3,
    # we assign the missing values in Embarked to 'S':
    if len(df[df['Embarked'].isnull()] > 0):
        df.fillna('S')

    # Fill in missing values of Fare with the average Fare
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({None: avg_fare}, inplace=True)

    # To keep Age in tact, make a copy of it called AgeFill
    # that we will use to fill in the missing ages:
    df['AgeFill'] = df['Age']

    # Determine the Age typical for each passenger class by Sex_Val.
    # We'll use the median instead of the mean because the Age
    # histogram seems to be right skewed.
    df['AgeFill'] = df['AgeFill'] \
        .groupby([df['Sex_Val'], df['Pclass']]) \
        .apply(lambda x: x.fillna(x.median()))

    # Define a new feature FamilySize that is the sum of
    # Parch (number of parents or children on board) and
    # SibSp (number of siblings or spouses):
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # Drop the columns we won't use:
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # Drop the Age column since we will be using the AgeFill column instead.
    # Drop the SibSp and Parch columns since we will be using FamilySize.
    # Drop the PassengerId column since it won't be used as a feature.
    df = df.drop(['Age', 'SibSp', 'Parch'], axis=1)

    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)

    return df

# read data
df_train = pd.read_csv('train.csv')
df_train = clean_data(df_train, drop_passenger_id=True)
train_data = df_train.values
clf = RandomForestClassifier(n_estimators=100)
train_features = train_data[:, 1:]
train_target = train_data[:, 0]
clf = clf.fit(train_features, train_target)

df_test = pd.read_csv('test.csv')
df_test = clean_data(df_test, drop_passenger_id=False)
test_data = df_test.values
test_x = test_data[:, 1:]
test_y = clf.predict(test_x)

df_test['Survived'] = test_y
df_test['Survived'] = df_test['Survived'].astype(int)
df_test[['PassengerId', 'Survived']].to_csv('results-rf.csv', index=False)
