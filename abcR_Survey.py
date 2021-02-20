import os
import sys
import pandas as pd
from util import runABC, writeABC, extract_data

def visualize(df, opt):
    print(df)
    hA = df.to_html()
    fA = open("./Survey/Reinforced_Survey"+ opt +".html", "w")
    fA.write(hA)
    fA.close()

if __name__ == "__main__":

    opt = "_1_0_with_balance"
    
    print("Option : ", opt[1:],"\n\n")

    if os.path.exists("./Survey/Reinforced_Survey"+opt+".pkl"):
        df = pd.read_pickle("./Survey/Reinforced_Survey"+opt+".pkl")
    else:
        df = pd.DataFrame(columns=["Benchmark","ABC_Area","Reinforced_Area","ABC_Delay","Reinforced_Delay"])

    dir = "./results"
    bench_dir = "./bench"

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
                    df.to_pickle("./Survey/Reinforced_Survey"+opt+".pkl")
                    # df_delay.to_pickle("Reinforced_Survey_Delay"+opt+".pkl")            

    visualize(df, opt)
    print("\n\nOption :", opt[1:])

