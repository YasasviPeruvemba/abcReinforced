import pandas as pd
import os

dir = "./inference_bench/"
survey_dir = "./inference_survey/"

options = ["2_9_without_balance", "2_9_with_balance", "2_7_without_balance", "2_7_with_balance", "2_5_without_balance", "2_5_with_balance", "2_3_without_balance", "2_3_with_balance", "1_1_without_balance", "1_1_with_balance", "1_0_without_balance", "1_0_with_balance", "0_1_without_balance", "0_1_with_balance"]

df_c = pd.DataFrame(columns=["Benchmark","ABC_Area","Reinforced_Area","ABC_Delay","Reinforced_Delay","Option"])

class Pair:
    def __init__(self, delay, area, opt):
        self.delay = float(delay)
        self.area = float(area)
        self.opt = opt
    def __lt__(self, other):
        if (float(self.delay) == float(other.delay)):
            return self.area < other.area
        else:
            return self.delay < other.delay
    def __eq__(self, other):
        return float(self.area) == float(other.area) and float(self.delay) == float(other.delay)

for subdir, dirs, files in os.walk(dir,topdown=True):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".aig"):
                vals = []
                for opt in options:
                    df = pd.read_pickle("./inference_survey/Reinforced_Survey_"+opt+".pkl")
                    row = df.loc[df["Benchmark"]==file]
                    abc_area = float(row.iloc[0,1])
                    abc_delay = float(row.iloc[0,3])
                    vals.append(Pair(float(row.iloc[0,4]), float(row.iloc[0,2]), opt))
                vals = sorted(vals)
                for i in range(3):
                    r = vals[i]
                    df_c.loc[0 if pd.isnull(df_c.index.max()) else df_c.index.max() + 1] = [file, abc_area, r.area, abc_delay, r.delay, r.opt]

df_c.to_pickle("./inference_survey/Consolidated_Reinforced_Survey.pkl")
print(df_c)
