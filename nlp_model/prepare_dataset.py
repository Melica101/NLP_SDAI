import pandas as pd

fake = pd.read_csv("data/fake.csv")
true = pd.read_csv("data/true.csv")

fake["label"] = 1
true["label"] = 0

data = pd.concat([fake[["title", "label"]], true[["title", "label"]]])
data = data.sample(frac=1).reset_index(drop=True)

data.to_csv("data/merged_data.csv", index=False)
print("Saved dataset to data/merged_data.csv")
