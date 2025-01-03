# utility script to convert CSV into Cerebrum format

ifile = "/home/james/data/chess/labeled/positions.csv"
ofile = "/home/james/data/chess/labeled/positions-cerebrum.csv"

with open(ifile, "r") as infile, open(ofile, "w") as outfile:
    for line in infile:
        [score, rest] = line.split(',')
        epd = rest.split(' ')
        wscore = float(score) / 100.0
        if 'b' == epd[1]:
            wscore = -wscore
        elif 'w' != epd[1]:
            raise Exception("unknown ptm!")
        outline = epd[0] + ';' + str(wscore) + ';' + str(-wscore) + '\n'
        outfile.write(outline)
