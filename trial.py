import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
from xgboost import XGBClassifier

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("Results.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)

df.dropna(inplace=True)
df.drop(["neutral"], axis=1, inplace=True)

df["match_id"] = df.index

# ----------------------------
# TEAM MAPPING
# ----------------------------
teams = pd.concat([df["home_team"], df["away_team"]]).unique()
team_mapping = {team: i for i, team in enumerate(teams)}

df["HomeTeamID"] = df["home_team"].map(team_mapping)
df["AwayTeamID"] = df["away_team"].map(team_mapping)

# ----------------------------
# ELO SYSTEM
# ----------------------------
elo = {team: 1500 for team in team_mapping.values()}

def expected_score(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))

home_elo, away_elo = [], []
K = 20

for _, row in df.iterrows():
    h = row["HomeTeamID"]
    a = row["AwayTeamID"]

    h_rating = elo[h]
    a_rating = elo[a]

    home_elo.append(h_rating)
    away_elo.append(a_rating)

    exp_h = expected_score(h_rating, a_rating)
    exp_a = 1 - exp_h

    if row["home_score"] > row["away_score"]:
        act_h, act_a = 1, 0
    elif row["home_score"] < row["away_score"]:
        act_h, act_a = 0, 1
    else:
        act_h, act_a = 0.5, 0.5

    elo[h] += K * (act_h - exp_h)
    elo[a] += K * (act_a - exp_a)

df["HomeELO"] = home_elo
df["AwayELO"] = away_elo
df["EloDiff"] = df["HomeELO"] - df["AwayELO"]

# ----------------------------
# TARGET
# ----------------------------
df["FTR"] = np.select(
    [
        df["home_score"] > df["away_score"],
        df["home_score"] < df["away_score"]
    ],
    [0, 2],
    default=1
)

# ----------------------------
# BETTER FORM FEATURES (IMPORTANT FIX)
# ----------------------------
df["home_points"] = df["FTR"].map({0: 3, 1: 1, 2: 0})
df["away_points"] = df["FTR"].map({0: 0, 1: 1, 2: 3})

df["home_form"] = df.groupby("home_team")["home_points"].transform(
    lambda x: x.shift(1).rolling(5).mean()
)

df["away_form"] = df.groupby("away_team")["away_points"].transform(
    lambda x: x.shift(1).rolling(5).mean()
)

# ----------------------------
# GOALS FORM
# ----------------------------
df["home_goals_5"] = df.groupby("home_team")["home_score"].transform(
    lambda x: x.shift(1).rolling(5).mean()
)

df["away_goals_5"] = df.groupby("away_team")["away_score"].transform(
    lambda x: x.shift(1).rolling(5).mean()
)

# ----------------------------
# NEW HIGH-VALUE FEATURES
# ----------------------------

# strength gap
df["elo_abs_diff"] = abs(df["EloDiff"])

# ratio strength (important)
df["elo_ratio"] = df["HomeELO"] / (df["AwayELO"] + 1)

# draw volatility signal (VERY IMPORTANT)
df["volatility"] = df["EloDiff"].rolling(5).std()

# low scoring expectation (proxy for draw likelihood)
df["low_score_env"] = df["home_goals_5"] + df["away_goals_5"]

# ----------------------------
# CLEAN DATA
# ----------------------------
df = df.fillna(0)

# ----------------------------
# FEATURES
# ----------------------------
X = df[[
    "HomeELO",
    "AwayELO",
    "EloDiff",
    "elo_abs_diff",
    "elo_ratio",
    "home_form",
    "away_form",
    "home_goals_5",
    "away_goals_5",
    "volatility",
    "low_score_env"
]]

Y = df["FTR"]

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.25,
    random_state=42,
    shuffle=True
)

# ----------------------------
# MODEL (TUNED XGBOOST)
# ----------------------------
model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# PROBABILITY OUTPUT (IMPORTANT)
# ----------------------------
proba = model.predict_proba(X_test)
predictions = np.argmax(proba, axis=1)

# ----------------------------
# EVALUATION
# ----------------------------
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

print("\nLog Loss:", log_loss(y_test, proba))