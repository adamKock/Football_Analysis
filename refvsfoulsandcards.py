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
df.drop(["FTHG","FTAG","FTR","HTHG","HTAG","HTR","HS","AS","HST","AST","HC","AC"] ,axis='columns', inplace=True)


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
ref_names={
    "A Taylor":21,
    "C Pawson":22,
    "S Barrott":23,
    "R Jones":24,
    "M Oliver":25,
    "J Gillett":26,
    "D England":27,
    "P Bankes":28,
    "S Hooper":29,
    "C Kavanagh":30,
    "T Bramall":31,
    "T Harrington":32,
    "M Salisbury":33,
    "S Attwell":34,
    "A Madley":35,
    "J Brooks":36,
    "A Kitchen":37,
    "T Robinson":38,
    "T Kirk":39,
    "P Tierney":40,
    "M Donohue":41,
    "F Hallam":42,
    "L Smith":43

}
df["Referee"]=df["Referee"].map(ref_names)

df["HomeTeamID"]=df["HomeTeam"].map(team_names)
df["AwayTeamID"]=df["AwayTeam"].map(team_names)

df.drop(["HomeTeam","AwayTeam"] , axis='columns', inplace=True)

teams = pd.concat([df["HomeTeamID"], df["AwayTeamID"]]).unique()

#What are we trying to see,
#if there is a relationships between referees and givign fouls/cards to their favourite team 
#Trying to spot if referees favour teams for fouls and card games. 

#Cols we need are ref HF AF	HY	AY	HR	AR
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date")

home_stats_cols = [
    "HF", "HY" ,"HR","HTI"
]

away_stats_cols = [
    "AF", "AY","AR","ATI"
]
print(df.head(10))

df[[col + "_avg5" for col in home_stats_cols]] = (
    df.groupby("HomeTeamID")[home_stats_cols]
    .transform(lambda x: x.shift(1).rolling(5).mean())
)

df[[col + "_avg5" for col in away_stats_cols]] = (
    df.groupby("AwayTeamID")[away_stats_cols]
    .transform(lambda x: x.shift(1).rolling(5).mean())
)

X = df[[
    "Referee",
    "HF_avg5",
    "HY_avg5",
    "HR_avg5",
    "HTI_avg5",
    "AF_avg5",
    "AY_avg5",
    "AR_avg5",
    "ATI_avg5" 
]]

Y = df[["FACH"]]


x_train, x_test, y_train, y_test = train_test_split(X,Y, 
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)


#model = RandomForestRegressor(random_state=42)

model = RandomForestClassifier(random_state=42)

model.fit(x_train,y_train)

predictions = model.predict(x_test)


accuracy_score = accuracy_score(y_test, predictions)
#mae = mean_absolute_error(y_test, predictions)

print("Accuracy:", accuracy_score)
#print("MAE:", mae)
