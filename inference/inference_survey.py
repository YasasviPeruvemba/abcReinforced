import os
import sys
import pandas as pd
from util_inference import runABC, writeABC, extract_data

def visualize(df, opt):
    print(df)
    hA = df.to_html()
    fA = open("./inference_survey/" + opt[5:] + "/Reinforced_Survey"+ opt +".html", "w")
    fA.write(hA)
    fA.close()

def reinforced_survey(option, coef):

    opt = "_" + coef + "_" + option
    print("Option : ", opt[1:],"\n\n")

    if not os.path.exists("./inference_survey/" + option):
        os.system("mkdir ./inference_survey/" + option)

    if os.path.exists("./inference_survey/" + option + "/Reinforced_Survey"+opt+".pkl"):
        df = pd.read_pickle("./inference_survey/" + option + "/Reinforced_Survey"+opt+".pkl")
    else:
        df = pd.DataFrame(columns=["Benchmark","ABC_Area","Reinforced_Area","ABC_Delay","Reinforced_Delay"])

    dir = "./inference_results/" + option
    bench_dir = "./inference_bench"

    for subdir, dirs, files in os.walk(bench_dir,topdown=True):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".aig"):
                    path = dir + os.sep + file[0:-4] + opt + ".csv"
                    f = open(path, "r")
                    command = f.read().splitlines()[-2]
                    steps = len(command.split(";"))

                    print("Running ABC on ....", file)
                    # Run compress2rs on ABC and accumulate stats
                    writeABC(filepath, command, opt=0)
                    runABC()
                    c_area, c_delay = extract_data()
                    # Run custom sequence on ABC and accumulate stats
                    print("Custom Command\n", command + "\n")
                    writeABC(filepath, command, opt=1)
                    runABC()
                    r_area, r_delay = extract_data()
                    # Now aggregate the stats
                    df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [file, c_area, r_area, c_delay, r_delay]
                    # Update the pickle files
                    df.to_pickle("./inference_survey/" + option + "/Reinforced_Survey"+opt+".pkl")
                    # df_delay.to_pickle("Reinforced_Survey_Delay"+opt+".pkl")            

    visualize(df, opt)
    print("\n\nOption :", opt[1:])