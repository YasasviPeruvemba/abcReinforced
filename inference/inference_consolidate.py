import pandas as pd
import os

dir = "./inference_bench/"

options = ["2_9_without_balance", "2_9_with_balance", "2_7_without_balance", "2_7_with_balance", "2_5_without_balance", "2_5_with_balance", "2_3_without_balance", "2_3_with_balance", "1_1_without_balance", "1_1_with_balance", "1_0_without_balance", "1_0_with_balance", "0_1_without_balance", "0_1_with_balance"]

df_c = pd.DataFrame(columns=["Benchmark","ABC_Area","Reinforced_Area","ABC_Delay","Reinforced_Delay","Option"])

for subdir, dirs, files in os.walk(dir,topdown=True):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".aig"):
                min_delay = 100000000
                min_area = 10000000
                abc_delay = None
                abc_area = None
                option = None
                for opt in options:
                    df = pd.read_pickle("./inference_survey/Reinforced_Survey_"+opt+".pkl")
                    row = df.loc[df["Benchmark"]==file]
                    #print(row)
                    if opt != "2_9_without_balance":
                        assert(abc_area == row.iloc[0,1])
                        assert(abc_delay == row.iloc[0,3])
                    abc_area = float(row.iloc[0,1])
                    abc_delay = float(row.iloc[0,3])
                    if float(row.iloc[0,4]) < min_delay :
                        min_delay = float(row.iloc[0,4])
                        min_area = float(row.iloc[0,2])
                        option = opt
                    elif float(row.iloc[0,4]) == min_delay:
                        if float(row.iloc[0,2]) < min_area:
                            min_delay = float(row.iloc[0,4])
                            min_area = float(row.iloc[0,2])
                            option = opt
                df_c.loc[0 if pd.isnull(df_c.index.max()) else df_c.index.max() + 1] = [file, abc_area, min_area, abc_delay, min_delay, option]

df_c.to_pickle("./inference_survey/Reinforced_Survey_Consolidated.pkl")
print(df_c)
