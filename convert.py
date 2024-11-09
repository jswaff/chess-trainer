# convert CSV file for testing
import os

in_fname = '/home/james/data/chess/labeled/bfd-d0-1.csv'
out_fname = 'bfd-d0-1.txt'

with open(in_fname, 'r') as in_file:
    with open(out_fname, 'w') as out_file:
        for line in in_file:
            #print(line)
            (score,fen) = line.split(",",1)
            # 8/1k4B1/7p/1pb2K2/2p1r3/8/6R1/8 b - - 1 56
            fenparts = fen.split(" ")
            pos = fenparts[0]
            stm = fenparts[1]
            if stm == 'w':
                w_score = int(score)
            elif stm == 'b':
                w_score = 0-int(score)
            else:
                raise Exception(f'invalid stm {stm}')
            outline = "{};{};{}\n".format(pos,w_score,-w_score)
            out_file.write(outline)

