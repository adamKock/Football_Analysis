import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

home_teams = []
away_teams=[]

df = pd.read_csv("Results.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date")
home_teams = df["home_team"].to_list()
away_teams = df["away_team"].to_list()

total_teams = home_teams + away_teams 

all_teams = list(dict.fromkeys(total_teams))
team_mapping={}
for i, team in enumerate(all_teams):
    team_mapping.update({
        team:i
    })

df.dropna(how="any", inplace=True, axis=0)
df.drop(["neutral"], axis=1, inplace=True)



df["HomeTeamID"]=df["home_team"].map(team_mapping)
df["AwayTeamID"]=df["away_team"].map(team_mapping)

print(df.head(10))

teams = pd.concat([df["HomeTeamID"], df["AwayTeamID"]]).unique()

elo = {team: 1500 for team in teams}
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

home_elo = []
away_elo = []

K = 20

for _, row in df.iterrows():
    home = row["HomeTeamID"]
    away = row["AwayTeamID"]

    home_rating = elo[home]
    away_rating = elo[away]
   

    # store ratings BEFORE the match
    home_elo.append(home_rating)
    away_elo.append(away_rating)

    # expected outcome
    exp_home = expected_score(home_rating, away_rating)
    exp_away = 1 - exp_home

    # actual result
    if row["home_score"] > row["away_score"]:
        actual_home = 1
        actual_away = 0
    elif row["home_score"] < row["away_score"]:
        actual_home = 0
        actual_away = 1
    else:
        actual_home = 0.5
        actual_away = 0.5

    # update ratings
    elo[home] += K * (actual_home - exp_home)
    elo[away] += K * (actual_away - exp_away)



df["HomeELO"] = home_elo
df["AwayELO"] = away_elo

df["FTR"] = None  # default

df.loc[df["home_score"] > df["away_score"], "FTR"] = "H"
df.loc[df["home_score"] < df["away_score"], "FTR"] = "A"
df.loc[df["home_score"] == df["away_score"], "FTR"] = "D"

df["FTR"] = df["FTR"].map({"H":0,"D":1,"A":2})
df["home_win"] = (df["FTR"] == 0).astype(int)
df["away_win"] = (df["FTR"] == 2).astype(int)
df["draw"] = (df["FTR"] == 1).astype(int)

df["EloDiff"] = df["HomeELO"] - df["AwayELO"]

#Now we need to make a rolling averages for the last 5 games of each team, 
#Creating it for form, goals scored and goals conceded for all teams in the dataset

df["home_form"] = df.groupby("home_team")["FTR"].transform(lambda x: x.shift(1).rolling(5).mean())
df["away_form"] = df.groupby("away_team")["FTR"].transform(lambda x: x.shift(1).rolling(5).mean())


df["home_avg_goals_5"] = df.groupby("home_team")["home_score"].transform(lambda x: x.shift(1).rolling(5).mean())
df["away_avg_goals_5"] = df.groupby("away_team")["away_score"].transform(lambda x: x.shift(1).rolling(5).mean())




X = df[[
    "HomeELO",
    "AwayELO",
    "EloDiff",
    "home_form",
    "away_form",
    "home_avg_goals_5",
    "away_avg_goals_5"
]]

Y=df["FTR"]

x_train, x_test, y_train, y_test = train_test_split(X,Y, 
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)



model = RandomForestClassifier(random_state=42)

model.fit(x_train,y_train)

predictions = model.predict(x_test)


acc= accuracy_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("Accuracy:", acc)
print("MAE:", mae)
                                   




