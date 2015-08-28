# test.py
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def load_file():
    train_df = pd.read_csv("train.csv", parse_dates=["Dates"])
    train_df["Hour"] = train_df["Dates"].map(lambda x: x.hour)

    train_df["DayOfWeek"] = pd.Categorical.from_array(
        train_df['DayOfWeek']).codes
    train_df["PdDistrict"] = pd.Categorical.from_array(
        train_df['PdDistrict']).codes

    cols = ["DayOfWeek", "PdDistrict", "X", "Y"]
    data = train_df[cols]
    category = pd.Categorical.from_array(train_df["Category"])

    return data, category


def func_category(cat):
    if str(cat) == "WEAPON LAWS":
        return 1
    elif str(cat) == "WARRANTS":
        return 2
    elif str(cat) == "VEHICLE THEFT":
        return 3
    elif str(cat) == "VANDALISM":
        return 4
    elif str(cat) == "TRESPASS":
        return 5
    elif str(cat) == "TREA":
        return 6
    elif str(cat) == "SUSPICIOUS OCC":
        return 7
    elif str(cat) == "SUICIDE":
        return 8
    elif str(cat) == "STOLEN PROPERTY":
        return 9
    elif str(cat) == "SEX OFFENSES FORCIBLE":
        return 10
    elif str(cat) == "SEX OFFENSES NON FORCIBLE":
        return 11
    elif str(cat) == "SECONDARY CODES":
        return 12
    elif str(cat) == "RUNAWAY":
        return 13
    elif str(cat) == "ROBBERY":
        return 14
    elif str(cat) == "RECOVERED VEHICLE":
        return 15
    elif str(cat) == "PROSTITUTION":
        return 16
    elif str(cat) == "PORNOGRAPHY/OBSCENE MAT":
        return 17
    elif str(cat) == "OTHER OFFENSES":
        return 18
    elif str(cat) == "NON-CRIMINAL":
        return 19
    elif str(cat) == "MISSING PERSON":
        return 20
    elif str(cat) == "LOITERING":
        return 21
    elif str(cat) == "LIQUOR LAWS":
        return 22
    elif str(cat) == "LARCENY/THEFT":
        return 23
    elif str(cat) == "KIDNAPPING":
        return 24
    elif str(cat) == "GAMBLING":
        return 25
    elif str(cat) == "FRAUD":
        return 26
    elif str(cat) == "FORGERY/COUNTERFEITING":
        return 27
    elif str(cat) == "FAMILY OFFENSES":
        return 28
    elif str(cat) == "EXTORTION":
        return 29
    elif str(cat) == "EMBEZZLEMENT":
        return 30
    elif str(cat) == "DRUNKENNESS":
        return 31
    elif str(cat) == "DRUG/NARCOTIC":
        return 32
    elif str(cat) == "DRIVING UNDER THE INFLUENCE":
        return 33
    elif str(cat) == "DISORDERLY CONDUCT":
        return 34
    elif str(cat) == "BURGLARY":
        return 35
    elif str(cat) == "BRIBERY":
        return 36
    elif str(cat) == "BAD CHECKS":
        return 37
    elif str(cat) == "ASSAULT":
        return 38
    elif str(cat) == "ARSON":
        return 39
    else:
        return 0


def func_PdDistrict(dist):
    if str(dist) == "SOUTHERN":
        return 1
    elif str(dist) == "MISSION":
        return 2
    elif str(dist) == "NORTHERN":
        return 3
    elif str(dist) == "BAYVIEW":
        return 4
    elif str(dist) == "CENTRAL":
        return 5
    elif str(dist) == "TENDERLOIN":
        return 6
    elif str(dist) == "INGLESIDE":
        return 7
    elif str(dist) == "TARAVAL":
        return 8
    elif str(dist) == "PARK":
        return 9
    elif str(dist) == "RICHMOND":
        return 10
    else:
        return 0


def func_dayofweek(weekday):
    if str(weekday) == "Monday":
        return 1
    elif str(weekday) == "Tuesday":
        return 2
    elif str(weekday) == "Wednesday":
        return 3
    elif str(weekday) == "Thursday":
        return 4
    elif str(weekday) == "Friday":
        return 5
    elif str(weekday) == "Saturday":
        return 6
    elif str(weekday) == "Sunday":
        return 7
    else:
        return 0


def plot_event_hourly(train_df):
    train_df["Hour"] = train_df["Dates"].map(lambda x: x.hour)
    train_df["event"] = 1
    hourly_event = train_df[["Hour", "event"]].groupby(
        ["Hour"]).count().reset_index()
    hourly_event.plot(kind="bar")
    plt.show()


def plot_event_two_wk(train_df):
    train_df["Year"] = train_df["Dates"].map(lambda x: x.year)
    train_df["Week"] = train_df["Dates"].map(lambda x: x.week)

    train_df["event"] = 1
    weekly_event = train_df[["Week", "Year", "event"]].groupby(
        ["Year", "Week"]).count().reset_index()
    weekly_event_years = weekly_event.pivot(
        index="Week", columns="Year", values="event").fillna(method="ffill")
    weekly_event_years.interpolate().plot()
    plt.show()


# to check crime count by district
def plot_District(train_df):
    dist_count = train_df.PdDistrict.value_counts()
    plt.figure()
    dist_count.plot(kind="barh")
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()


# to check crime count by day of week
def plot_Dayofweek(train_df):
    day_count = train_df.groupby("DayOfWeek").count()
    plt.figure()
    day_count.sort(columns="Dates", ascending=1)["Dates"].plot(kind="barh")
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()


# plot crime by category - to check the count of the category of crime
def plot_Category(train_df):
    cat_Count = train_df.groupby("Category").count()
    plt.figure()
    cat_Count.sort(columns="Dates", ascending=1)["Dates"].plot(kind="barh")
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()


def learn_model(data, category):
    data_train, data_test, target_train, target_test = cross_validation.train_test_split(
        data, category, test_size=0.4, random_state=43)
    classifier = BernoulliNB().fit(data_train, target_train)
    predicted = classifier.predict(data_test)
    evaluate_model(target_test, predicted)


def evaluate_model(target_true, target_predicted):
    print(confusion_matrix(target_true, target_predicted))
    print(accuracy_score(target_true, target_predicted))

# definition of main function
data, category = load_file()
learn_model(data, category)
