import os

# This is used to extract the number of nodes and level of the AIG
def writeABC_map(filepath, cmd, opt=0):
    f = open("run.txt", "w")
    f.write("read ")
    f.write(filepath)
    f.write("\n")
    f.write("strash")
    f.write("\n")
    if opt == 1:
        f.write(cmd)    
    elif opt == 0:
        f.write("balance -l; resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; balance -l; resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; balance -l; resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l; balance -l")
    elif opt == 2:
        f.write("balance -l; resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; balance -l; resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; balance -l; resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l; balance -l; dch; balance -l;")
    elif opt == 3:
        f.write("dch; balance -l; resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; balance -l; resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; balance -l; resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l; balance -l;")
    elif opt == 4:
        f.write("resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l;")
    elif opt == 5:
        f.write("balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;")
    f.write("\n")
    f.write("print_stats")
    f.close()

# This is used to find the Area and Delay from either mcnc or 6-LUT mapping
def writeABC(filepath, cmd, opt=0, map=False):
    f = open("run.txt", "w")
    f.write("read_library ./abc/mcnc.genlib")
    f.write("\n")
    f.write("read ")
    f.write(filepath)
    f.write("\n")
    f.write("strash")
    f.write("\n")
    if opt == 1:
        f.write(cmd)    
    elif opt == 0:
        f.write("balance -l; resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; balance -l; resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; balance -l; resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l; balance -l")
    elif opt == 2:
        f.write("balance -l; resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; balance -l; resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; balance -l; resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l; balance -l; dch; balance -l;")
    elif opt == 3:
        f.write("dch; balance -l; resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; balance -l; resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; balance -l; resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l; balance -l;")
    elif opt == 4:
        f.write("resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l;")
    elif opt == 5:
        f.write("balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;")
    f.write("\n")
    if map: f.write("map")
    else: f.write("if -K 6 -X 1 -Y 1")
    f.write("\n")
    f.write("print_stats -l")
    f.close()

def extract_data(map=False):
    path = "survey_data.txt"
    f = open(path, "r")
    line = f.readline()
    words = line.strip().split(" ")
    words = [word for word in words if (word != '=' and word!= '')]
    nd = int(words[1])
    level = int(words[3])
    # Comment this if-else clause if using writeABC_map
    if map:
        area = float(words[5])
        delay = float(words[7])
        return area, delay
    else:
        level = float(words[5])
    return nd, level

def runABC():
    cmd = "./abc/abc -f run.txt > survey.log"
    os.system(cmd)


def convertToAIG(filepath, filename):
    print(filename)
    f = open("run.txt", "w")
    f.write("read ")
    f.write(filepath)
    f.write("\n")
    f.write("strash")
    f.write("\n")
    f.write("write_aiger ")
    f.write("./bench/extra/" + filename + ".aig")
    f.write("\n")
    f.close()
    runABC()