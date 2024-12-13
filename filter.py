# utility script to filter out mate scores

ifile = "/home/james/data/chess/labeled/positions-hce.csv"
ofile = "/home/james/data/chess/labeled/positions-hce-filtered.csv"

with open(ifile, "r") as infile, open(ofile, "w") as outfile:
    for line in infile:
        [score, rest] = line.split(',')
        if abs(int(score)) < 32000:
            outfile.write(line)
