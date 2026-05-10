import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score


#What we are trying to predict for the up and coming game 
#Total shots
#Total fouls 
#Total corners
#Total cards 


#Import and clean the data 
df = pd.read_csv("Prem.csv")
df = df.drop("Div" ,axis='columns')


df.drop(["HTR","Referee","HTHG","HTAG"] , axis='columns', inplace=True)


team_names={
    "Liverpool":1,
    "Aston Villa":2,
    "Brighton":3,    
    "Sunderland":4,
    "Tottenham":5,
    "Wolves":6,    
    "Chelsea":7,        
    "Nott'm Forest":8,
    "Man United":9,
    "Leeds":10,
    "West Ham":11,
    "Man City":12,
    "Bournemouth":13,
    "Brentford":14,
    "Burnley":15,
    "Arsenal":16,
    "Crystal Palace":17,
    "Everton":18,
    "Fulham":19,
    "Newcastle":20
}

df["HomeTeamID"]=df["HomeTeam"].map(team_names)
df["AwayTeamID"]=df["AwayTeam"].map(team_names)

df.drop(["HomeTeam","AwayTeam"] , axis='columns', inplace=True)

teams = pd.concat([df["HomeTeamID"], df["AwayTeamID"]]).unique()

elo = {team: 1500 for team in teams}

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

home_elo = []
away_elo = []

K = 20  # sensitivity factor

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
    if row["FTHG"] > row["FTAG"]:
        actual_home = 1
        actual_away = 0
    elif row["FTHG"] < row["FTAG"]:
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

home_stats_cols = [
    "HS", "HST" ,"HF", "HC", "HY", "HR"
]

away_stats_cols = [
    "AS", "AST","AF", "AC", "AY", "AR"
]

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date")

df["FTR"] = df["FTR"].map({"H":0,"D":1,"A":2})

df[[col + "_avg5" for col in home_stats_cols]] = (
    df.groupby("HomeTeamID")[home_stats_cols]
    .transform(lambda x: x.shift(1).rolling(5).mean())
)

df[[col + "_avg5" for col in away_stats_cols]] = (
    df.groupby("AwayTeamID")[away_stats_cols]
    .transform(lambda x: x.shift(1).rolling(5).mean())
)
    

df = df.dropna()

X = df[[
    "HS_avg5",
    "HF_avg5",
    "HC_avg5",
    "HY_avg5",
    "HomeELO",
    "AwayELO",
    "AS_avg5",
    "AF_avg5",
    "AC_avg5",
    "AY_avg5",  
]]

Y = df[["FTR"]]



x_train, x_test, y_train, y_test = train_test_split(X,Y, 
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)


#model = RandomForestRegressor(random_state=42)

model = RandomForestClassifier(random_state=42)

model.fit(x_train,y_train)

predictions = model.predict(x_test)


acc= accuracy_score(y_test, predictions)
#mae = mean_absolute_error(y_test, predictions)

print("Accuracy:", acc)
#print("MAE:", mae)
print(df["HomeELO"])
                                   




