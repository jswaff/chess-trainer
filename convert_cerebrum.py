# utility script to convert CSV into Cerebrum format

ifile = "/home/james/data/chess/labeled/positions-d3.csv"
ofile = "/home/james/data/chess/labeled/positions-d3-cerebrum.csv"

with open(ifile, "r") as infile, open(ofile, "w") as outfile:
    for line in infile:
        [score, rest] = line.split(',')
        epd = rest.split(' ')
        wscore = int(score)
        if 'b' == epd[1]:
            wscore = -wscore
        elif 'w' != epd[1]:
            raise Exception("unknown ptm!")
        outline = epd[0] + ';' + str(wscore) + ';' + str(-wscore) + '\n'
        outfile.write(outline)
