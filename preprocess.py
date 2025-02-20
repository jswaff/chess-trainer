# utility script for preprocessing

ifile = "/home/james/data/chess/labeled/positions-d5.csv"
ofile = "/home/james/data/chess/labeled/positions-d5-preprocessed.csv"

with open(ifile, "r") as infile, open(ofile, "w") as outfile:
    for line in infile:
        [score, rest] = line.split(',')
        iscore = int(score)
        if abs(iscore) < 32000: # omit mates
            # clamp at +/- 15 pawns
            if iscore < -1500:
                iscore = -1500
            elif iscore > 1500:
                iscore = 1500
            outline = str(iscore) + ',' + rest
            outfile.write(outline)
