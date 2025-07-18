# utility script for clamping scores

ifile = "data/positions.csv"
ofile = "data/positions-clamped.csv"

with open(ifile, "r") as infile, open(ofile, "w") as outfile:
    for line in infile:
        line = line.strip()
        [fen,score,wins,draws,losses] = line.split(',')
        iscore = int(score)
        if abs(iscore) < 32000: # omit mates
            # clamp at +/- 15 pawns
            if iscore < -1500:
                iscore = -1500
            elif iscore > 1500:
                iscore = 1500
            outline = fen + ',' + str(iscore) + ',' + str(wins) + ',' + str(draws) + ',' + str(losses) + '\n'
            outfile.write(outline)
