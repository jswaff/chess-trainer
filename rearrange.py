# utility script for re-arranging a data file from format e0,e1,e2 to e2,e0,e1

ifile = "data/positions.csv"
ofile = "data/positions-out.csv"

with open(ifile, "r") as infile, open(ofile, "w") as outfile:
    for line in infile:
        line = line.strip()
        [e0,e1,e2] = line.split(',')
        outline = e2 + ',' + e0 + ',' + e1 + '\n'
        outfile.write(outline)
