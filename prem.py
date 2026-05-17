import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss

# =========================
# LOAD DATA
# =========================
files = [
    "season1.csv","season2.csv","season3.csv","season4.csv",
    "season5.csv","season6.csv","season7.csv","season8.csv",
    "season9.csv","season10.csv","season11.csv","season12.csv",
    "season13.csv","season14.csv"
]

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.columns = df.columns.str.strip()

# =========================
# CLEAN
# =========================
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
df = df.sort_values("Date").reset_index(drop=True)

# =========================
# TARGET
# =========================
df["FTR_num"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})

# =========================
# FEATURES
# =========================
df["goal_diff_ht"] = df["HTHG"] - df["HTAG"]
df["shot_diff"] = df["HS"] - df["AS"]
df["sot_diff"] = df["HST"] - df["AST"]
df["corner_diff"] = df["HC"] - df["AC"]
df["foul_diff"] = df["HF"] - df["AF"]

df = df.sort_values("Date")

df["home_points"] = df["FTR_num"].map({0:3, 1:1, 2:0})
df["away_points"] = df["FTR_num"].map({0:0, 1:1, 2:3})

df["home_form"] = df.groupby("HomeTeam")["home_points"].transform(lambda x: x.shift(1).rolling(5).mean())
df["away_form"] = df.groupby("AwayTeam")["away_points"].transform(lambda x: x.shift(1).rolling(5).mean())

df["form_diff"] = df["home_form"] - df["away_form"]

df["home_concede"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.shift(1).rolling(5).mean())
df["away_concede"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.shift(1).rolling(5).mean())
df["concede_diff"] = df["home_concede"] - df["away_concede"]

df["shot_acc_diff"] = (df["HST"]/(df["HS"]+1)) - (df["AST"]/(df["AS"]+1))

df = df.fillna(0)

# =========================
# FEATURES SET
# =========================
features = [
    "goal_diff_ht",
    "shot_diff",
    "sot_diff",
    "corner_diff",
    "foul_diff",
    "home_form",
    "away_form",
    "form_diff",
    "concede_diff",
    "shot_acc_diff"
]

X = df[features]
y = df["FTR_num"]

# =========================
# TIME SPLIT
# =========================
split = int(len(df) * 0.75)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================
# MODEL + CALIBRATION
# =========================
base_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
proba = model.predict_proba(X_test)
preds = np.argmax(proba, axis=1)

print("\n==================== MODEL ====================")
print("Accuracy:", accuracy_score(y_test, preds))
print("Log Loss:", log_loss(y_test, proba))
print(classification_report(y_test, preds))

# =========================
# BETTING DATA
# =========================
test_df = df.iloc[split:].copy()

test_df["odds_home"] = test_df["B365H"]
test_df["odds_draw"] = test_df["B365D"]
test_df["odds_away"] = test_df["B365A"]

# model probabilities
test_df["p_home"] = proba[:, 0]
test_df["p_draw"] = proba[:, 1]
test_df["p_away"] = proba[:, 2]

# =========================
# EV CALCULATION
# =========================
test_df["ev_home"] = (test_df["p_home"] * test_df["odds_home"]) - 1
test_df["ev_draw"] = (test_df["p_draw"] * test_df["odds_draw"]) - 1
test_df["ev_away"] = (test_df["p_away"] * test_df["odds_away"]) - 1

# =========================
# CLV CALCULATION (SIMPLIFIED)
# =========================
# proxy: compare model probability vs implied probability

test_df["imp_home"] = 1 / test_df["odds_home"]
test_df["imp_draw"] = 1 / test_df["odds_draw"]
test_df["imp_away"] = 1 / test_df["odds_away"]

total = test_df["imp_home"] + test_df["imp_draw"] + test_df["imp_away"]

test_df["imp_home"] /= total
test_df["imp_draw"] /= total
test_df["imp_away"] /= total

test_df["edge_home"] = test_df["p_home"] - test_df["imp_home"]
test_df["edge_draw"] = test_df["p_draw"] - test_df["imp_draw"]
test_df["edge_away"] = test_df["p_away"] - test_df["imp_away"]

# =========================
# BET FILTER (PRODUCTION LEVEL)
# =========================
MIN_EV = 0.06
MIN_EDGE = 0.03

bets = []

for i, row in test_df.iterrows():

    evs = {
        "H": row["ev_home"],
        "D": row["ev_draw"],
        "A": row["ev_away"]
    }

    edges = {
        "H": row["edge_home"],
        "D": row["edge_draw"],
        "A": row["edge_away"]
    }

    best_pick = max(evs, key=evs.get)
    best_ev = evs[best_pick]

    if best_ev > MIN_EV and edges[best_pick] > MIN_EDGE:
        bets.append((i, best_pick, best_ev))

# =========================
# KELLY STAKING
# =========================
def kelly(p, odds):
    b = odds - 1
    q = 1 - p
    return max((p * b - q) / b, 0)

bankroll = 1000
fraction = 0.25  # fractional Kelly (VERY IMPORTANT)
results = []

for i, pick, ev in bets:

    row = test_df.loc[i]

    if pick == "H":
        odds = row["odds_home"]
        p = row["p_home"]
        win = row["FTR_num"] == 0

    elif pick == "D":
        odds = row["odds_draw"]
        p = row["p_draw"]
        win = row["FTR_num"] == 1

    else:
        odds = row["odds_away"]
        p = row["p_away"]
        win = row["FTR_num"] == 2

    stake = bankroll * kelly(p, odds) * fraction
    stake = min(stake, bankroll * 0.02)  # cap risk per bet

    if stake <= 0:
        continue

    if win:
        bankroll += stake * (odds - 1)
        pnl = stake * (odds - 1)
    else:
        bankroll -= stake
        pnl = -stake

    results.append(pnl)

# =========================
# FINAL OUTPUT
# =========================
print("\n==================== BETTING ====================")
print("Total bets:", len(bets))
print("Final bankroll:", bankroll)
print("Profit:", bankroll - 1000)
print("ROI:", (bankroll - 1000) / 1000)