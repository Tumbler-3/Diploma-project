import pandas as pd  
import numpy as np
import json

df=pd.read_csv(r"full_df.csv")
df = df.drop(columns=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O', 'labels'])
targets = np.array(df["target"].apply(lambda x: json.loads(x)).tolist())

classes = { 0: "Normal",
            1: "Diabetes",
            2: "Glaucoma",
            3: "Cataract",
            4: "Age related Macular Degeneration",
            5: "Hypertension",
            6: "Pathological Myopia",
            7: "Other diseases/abnormalities"
          }
 
df["class_name"] = np.argmax(targets, axis=1).tolist()
df["class_name"] = df["class_name"].replace(classes)

df["class"] = np.argmax(targets, axis=1).tolist()
# print(df.head())

processed_data = df[["ID", "filename", 'class', "class_name", "target"]]
# print(processed_data.head())

processed_data.to_csv('df_proc.csv', index=False)
