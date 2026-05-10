import kagglehub
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATA
# =========================

path = kagglehub.dataset_download("nathanlauga/nba-games")

df = pd.read_csv(path + "/games.csv")
teams_df = pd.read_csv(path + "/teams.csv")

team_lookup = teams_df.set_index("TEAM_ID")["NICKNAME"]

df["HOME_TEAM_NAME"] = df["HOME_TEAM_ID"].map(team_lookup)
df["AWAY_TEAM_NAME"] = df["VISITOR_TEAM_ID"].map(team_lookup)

# =========================
# TARGET
# =========================

df["WINNER"] = np.where(
    df["PTS_home"] > df["PTS_away"],
    0,   # Home win
    1    # Away win
)

# =========================
# SORT BY DATE (CRITICAL)
# =========================

df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
df = df.sort_values("GAME_DATE_EST")

# =========================
# ELO SYSTEM
# =========================

INITIAL_ELO = 1500
K = 20
HOME_ADV = 100

elo = {}

home_elo = []
away_elo = []

for _, row in df.iterrows():

    h = row["HOME_TEAM_ID"]
    a = row["VISITOR_TEAM_ID"]

    if h not in elo:
        elo[h] = INITIAL_ELO
    if a not in elo:
        elo[a] = INITIAL_ELO

    h_elo = elo[h]
    a_elo = elo[a]

    home_elo.append(h_elo)
    away_elo.append(a_elo)

    exp_home = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADV)) / 400))

    if row["WINNER"] == 0:
        actual_home = 1
    else:
        actual_home = 0

    elo[h] = h_elo + K * (actual_home - exp_home)
    elo[a] = a_elo + K * ((1 - actual_home) - (1 - exp_home))

df["HOME_ELO"] = home_elo
df["AWAY_ELO"] = away_elo
df["ELO_DIFF"] = df["HOME_ELO"] - df["AWAY_ELO"]

# =========================
# ROLLING WIN %
# =========================

df["HOME_WIN"] = (df["WINNER"] == 0).astype(int)
df["AWAY_WIN"] = (df["WINNER"] == 1).astype(int)

df["HOME_WIN_PCT_10"] = (
    df.groupby("HOME_TEAM_ID")["HOME_WIN"]
    .transform(lambda x: x.shift(1).rolling(10).mean())
)

df["AWAY_WIN_PCT_10"] = (
    df.groupby("VISITOR_TEAM_ID")["AWAY_WIN"]
    .transform(lambda x: x.shift(1).rolling(10).mean())
)

# =========================
# BASIC TEAM STATS (NO LEAKAGE)
# =========================

home_cols = [
    "FG_PCT_home",
    "FT_PCT_home",
    "FG3_PCT_home",
    "AST_home",
    "REB_home"
]

away_cols = [
    "FG_PCT_away",
    "FT_PCT_away",
    "FG3_PCT_away",
    "AST_away",
    "REB_away"
]

df[[c + "_AVG5" for c in home_cols]] = (
    df.groupby("HOME_TEAM_ID")[home_cols]
    .transform(lambda x: x.shift(1).rolling(5).mean())
)

df[[c + "_AVG5" for c in away_cols]] = (
    df.groupby("VISITOR_TEAM_ID")[away_cols]
    .transform(lambda x: x.shift(1).rolling(5).mean())
)

# =========================
# SIMPLE OFF/DEF RATINGS (score-based)
# =========================

df["HOME_OFF_RTG"] = df["PTS_home"]
df["HOME_DEF_RTG"] = df["PTS_away"]

df["AWAY_OFF_RTG"] = df["PTS_away"]
df["AWAY_DEF_RTG"] = df["PTS_home"]

df["HOME_OFF_RTG_AVG10"] = (
    df.groupby("HOME_TEAM_ID")["HOME_OFF_RTG"]
    .transform(lambda x: x.shift(1).rolling(10).mean())
)

df["HOME_DEF_RTG_AVG10"] = (
    df.groupby("HOME_TEAM_ID")["HOME_DEF_RTG"]
    .transform(lambda x: x.shift(1).rolling(10).mean())
)

df["AWAY_OFF_RTG_AVG10"] = (
    df.groupby("VISITOR_TEAM_ID")["AWAY_OFF_RTG"]
    .transform(lambda x: x.shift(1).rolling(10).mean())
)

df["AWAY_DEF_RTG_AVG10"] = (
    df.groupby("VISITOR_TEAM_ID")["AWAY_DEF_RTG"]
    .transform(lambda x: x.shift(1).rolling(10).mean())
)

df["HOME_NET_RTG_AVG10"] = (
    df["HOME_OFF_RTG_AVG10"] - df["HOME_DEF_RTG_AVG10"]
)

df["AWAY_NET_RTG_AVG10"] = (
    df["AWAY_OFF_RTG_AVG10"] - df["AWAY_DEF_RTG_AVG10"]
)

# =========================
# CLEAN DATA
# =========================

df.dropna(inplace=True)

# =========================
# FEATURES
# =========================

X = df[[
    "FG_PCT_home_AVG5",
    "FT_PCT_home_AVG5",
    "FG3_PCT_home_AVG5",
    "AST_home_AVG5",
    "REB_home_AVG5",

    "FG_PCT_away_AVG5",
    "FT_PCT_away_AVG5",
    "FG3_PCT_away_AVG5",
    "AST_away_AVG5",
    "REB_away_AVG5",

    "HOME_ELO",
    "AWAY_ELO",
    "ELO_DIFF",

    "HOME_WIN_PCT_10",
    "AWAY_WIN_PCT_10",

    "HOME_OFF_RTG_AVG10",
    "HOME_DEF_RTG_AVG10",
    "HOME_NET_RTG_AVG10",

    "AWAY_OFF_RTG_AVG10",
    "AWAY_DEF_RTG_AVG10",
    "AWAY_NET_RTG_AVG10"
]]

Y = df["WINNER"]

# =========================
# CHRONOLOGICAL SPLIT
# =========================

split = int(len(df) * 0.75)

x_train = X.iloc[:split]
x_test = X.iloc[split:]

y_train = Y.iloc[:split]
y_test = Y.iloc[split:]

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(x_train, y_train)

# =========================
# PREDICTIONS
# =========================

preds = model.predict(x_test)
probs = model.predict_proba(x_test)

acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

# =========================
# SAMPLE OUTPUT
# =========================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": preds,
    "Home Win Prob": probs[:, 0],
    "Away Win Prob": probs[:, 1]
})

print(results.head(20))