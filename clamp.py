# utility script for clamping scores

import argparse

parser = argparse.ArgumentParser(description="Clamp score values in a chess position CSV file.")
parser.add_argument("ifile", help="Path to the input CSV file")
parser.add_argument("ofile", help="Path to the output CSV file")
args = parser.parse_args()

with open(args.ifile, "r") as infile, open(args.ofile, "w") as outfile:
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
