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

def expected_score(home_rating, away_rating):
    return 1 / (10 ** ((away_rating - home_rating) / 400) + 1)


home_elo = []
away_elo = []

K = 20  # sensitivity factor

for _, row in df.iterrows():
    #Defines who is the home team or away team
    home = row["HomeTeamID"]
    away = row["AwayTeamID"]

    #Assings the rating of the home and away. 
    # We go into the Elo dictionairy with the key (team) and get the value (elo rating)
    home_rating = elo[home]
    away_rating = elo[away]

    # store ratings BEFORE the match
    home_elo.append(home_rating)
    away_elo.append(away_rating)

    # expected outcome
    exp_home_win_pct = expected_score(home_rating, away_rating) 
    exp_away = 1 - exp_home_win_pct # This is the reverse EG if the exp home win is 60% then the exp away is 40%

    #Now we need to calculate the actuals and then update elo 

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
     #This works by getting the actual home result (1 or 0 or 0.5) and then minus the expected outcome
     #Then we do the same for the away team get the actual away result 1 or 0 or 0.5 and then minus the expected outcome
     #Then we times the outcome of that math operation by the sensitivity factor and then re assign that to the home and away team elo
    elo[home] += K * (actual_home - exp_home_win_pct)
    elo[away] += K * (actual_away - exp_away)

    #Total Steps for Elo System are 
    #Dictionairy to assign the teams the inital elo rating
    #Calculate the expected result for the home team and away team
    #Create Standard Elo Rating list for home and away
    #Initialise a sensitivity factor 
    #Loop through each of the rows assigning who the home and away teams are
    #Assigning  the elo rating for the home and away team
    #Appending that back to the elo rating list for home and away teams
    #Calculate the expected result for the home team and away team
    #Calculate the actual result for the home team and away team
    #Update the elo rating for the home team and away team
