# utility script to prepare a file for labeling
# the input should be an epd file containing ONLY a c1 opcode with the result
# e.g.  8/8/8/8/2k5/R7/7q/5K2 b - - c1 0-1;
# one way to produce that is using pgn-extract
# the following line would take every 10th position, and strips the c0 opcode:
# pgn-extract -Wepd test.pgn -s | awk 'NR % 10 == 0' | awk '{gsub(/c0.*c1/,"c1")}1' > test.epd
# the output is a file with lines with 0,<result>,fen
# e.g. 0,0,8/8/8/8/2k5/R7/7q/5K2 b - -
# where result is 1 for a white win, 0 for a black win, or 0.5 for a draw

ifile = "data/ccrl.epd"
ofile = "data/ccrl.csv"

with open(ifile, "r") as infile, open(ofile, "w") as outfile:
    for line in infile:
        [fen, result] = line.split('c1')
        fen = fen.strip()
        result = result.strip().replace(';','')
        if result == '1-0':
            iresult = 1.0
        elif result == '0-1':
            iresult = 0.0
        elif result == '1/2-1/2':
            iresult = 0.5
        else:
            print(f'skipping {fen} with result {result}')
            continue
        outline = '0,' + str(iresult) + ',' + fen + '\n'
        outfile.write(outline)

