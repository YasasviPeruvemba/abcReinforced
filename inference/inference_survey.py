import os
import sys
import pandas as pd
from util_inference import runABC, writeABC, extract_data

def visualize(df, opt):
    print(df)
    hA = df.to_html()
    fA = open("./inference_survey/Reinforced_Survey_"+ opt +".html", "w")
    fA.write(hA)
    fA.close()
    print("\n######################################################################\n")

options = ["2_9_without_balance", "2_9_with_balance", "2_7_without_balance", "2_7_with_balance", "2_5_without_balance", "2_5_with_balance", "1_0_without_balance", "1_0_with_balance", "0_1_without_balance", "0_1_with_balance"]

if __name__ == "__main__":

    dir = "./inference_results"
    bench_dir = "./inference_bench"
    for opt in options:
        #df = pd.DataFrame(columns=["Benchmark","ABC_Area","Reinforced_Area","ABC_Delay","Reinforced_Delay"])
        df = pd.read_pickle("./inference_survey/Reinforced_Survey_"+opt+".pkl")
        print("Option :",opt,"\n")
        for subdir, dirs, files in os.walk(bench_dir,topdown=True):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".aig"):
                    path = dir + os.sep + file[0:-4] + "_" + opt + ".csv"
                    #f = open(path, "r")
                    #command = f.read().splitlines()[-2]
                    #steps = len(command.split(";"))
                    #print("Running ABC on ....", file)
                    # Run compress2rs on ABC and accumulate stats
                    #writeABC(filepath, command, 0)
                    #runABC()
                    #c_area, c_delay = extract_data()
                    # Run custom sequence on ABC and accumulate stats
                    #print("Custom Command\n", command + "\n")
                    #writeABC(filepath, command, opt=1)
                    #runABC()
                    #r_area, r_delay = extract_data()
                    # Now aggregate the stats
                    #df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [file, c_area, r_area, c_delay, r_delay]
                    # Update the pickle files
                    #df.to_pickle("./inference_survey/Reinforced_Survey_"+opt+".pkl")
        visualize(df, opt)
