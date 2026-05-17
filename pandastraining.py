import pandas as pd 
import kagglehub

# Download latest version
path = kagglehub.dataset_download("heesoo37/120-years-of-olympic-history-athletes-and-results")

print("Path to dataset files:", path)

df = pd.read_csv(path + "/athlete_events.csv")


first_olympics = (df["Year"]==1896)

gold_medals_athens = (df["Medal"]=="Gold")& (df["City"]=="Athens")&df["Year"]==1896





 