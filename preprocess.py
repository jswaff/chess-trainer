# utility script for preprocessing

ifile = "/home/james/data/chess/labeled/all-d0.csv"
ofile = "/home/james/data/chess/labeled/all-d0-filtered.csv"

with open(ifile, "r") as infile, open(ofile, "w") as outfile:
    for line in infile:
        [score, rest] = line.split(',')
        iscore = int(score)
        # clamp at +/- 10 pawns
        if iscore < -1000:
            iscore = -1000
        elif iscore > 1000:
            iscore = 1000
        outline = str(iscore) + ',' + rest
        outfile.write(outline)
