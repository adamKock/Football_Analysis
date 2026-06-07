import pandas as pd 
import kagglehub

# Download latest version
path = kagglehub.dataset_download("heesoo37/120-years-of-olympic-history-athletes-and-results")

print("Path to dataset files:", path)


df = pd.read_csv(path + "/athlete_events.csv")
print(df.head(10))


first_olympics = (df["Year"]==1896)

gold_medals_athens = (df["Medal"]=="Gold")& (df["City"]=="Athens")&df["Year"]==1896

df["Name"].str.lower()

df["Name"].sort_values(ascending=True)

df.sort_values(by=["Year","Name"], ascending=[True,False])


print(df.columns)

df["Year"] = df["Year"].astype(int)



print(df.loc[df["Name"] == "Carl"])

print(df.info())

city = ["London","Athens", "Paris"]
start_dates=["1948","1896","1900"]

dy = pd.DataFrame({
    "City":city,
    "Start Date":start_dates
})

dx = pd.DataFrame(zip(city,start_dates),columns=["City","Start Date"])

start_time =["4th of August 2026", "28/08/1996", "12/08/2021"]

dx["Start Time"] = start_time
dx["Start Time"] = pd.to_datetime(dx["Start Time"], format='mixed')

print(dx)

end_dates=["3323","3434","2334"]


dt = pd.DataFrame({
    "City":city,
    "End Date":end_dates
})

print(pd.concat([dx,dt], axis=1))



print(pd.merge(left=dx, right=dt, how="inner", on=["City"]))





 