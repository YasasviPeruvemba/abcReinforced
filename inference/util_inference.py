import os

def writeABC(filepath, cmd, opt=0):
    f = open("run.txt", "w")
    f.write("read_library ../../ALS/abc/mcnc.genlib")
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
    f.write("\n")
    f.write("map")
    f.write("\n")
    f.write("print_stats")
    f.close()

def extract_data():
    path = "survey_data.txt"
    f = open(path, "r")
    line = f.readline()
    words = line.strip().split(" ")
    words = [word for word in words if (word != '=' and word!= '')]
    nd = int(words[1])
    edges = int(words[3])
    area = float(words[5])
    delay = float(words[7])
    return area,delay

def runABC():
    cmd = "../../ALS/abc/abc -f run.txt > survey.log"
    os.system(cmd)
