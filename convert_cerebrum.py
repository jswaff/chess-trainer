import os


with open("/home/james/data/chess/labeled/all-d1.csv", "r") as infile, open("positions-d1.txt", "w") as outfile:
    for line in infile:
        #print(line.rstrip("\n"))
        [score, rest] = line.split(',')
        epd = rest.split(' ')
        #print('\tscore: ', score)
        #print('\tpos: ', epd[0])
        #print('\tptm: ', epd[1])
        wscore = int(score)
        if 'b' == epd[1]:
            wscore = -wscore
        elif 'w' != epd[1]:
            raise Exception("unknown ptm!")
        if abs(wscore) > 32000:
            continue
        #print('\twscore: ', wscore)
        outline = epd[0] + ';' + str(wscore) + ';' + str(-1*wscore) + '\n'
        #print('\toutline: ', outline)
        outfile.write(outline)

