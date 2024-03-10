import numpy as np


class EPD:
    def toOHE(self, epd):
        epd_parts = epd.split(" ")
        ranks = epd_parts[0].split("/")
        ptm = epd_parts[1]

        ohe = np.zeros(768)
        sq = 0
        for r,rank in enumerate(ranks):
            for ch in rank:
                if '1' <= ch <= '8':
                    sq += int(ch)
                else:
                    if ch == 'R':
                        ohe[sq] = 1
                    elif ch=='r':
                        ohe[64+sq] = 1
                    elif ch == 'N':
                        ohe[128+sq] = 1
                    elif ch=='n':
                        ohe[192+sq] = 1
                    elif ch == 'B':
                        ohe[256+sq] = 1
                    elif ch=='b':
                        ohe[320+sq] = 1
                    elif ch == 'Q':
                        ohe[384+sq] = 1
                    elif ch=='q':
                        ohe[448+sq] = 1
                    elif ch == 'K':
                        ohe[512+sq] = 1
                    elif ch=='k':
                        ohe[576+sq] = 1
                    elif ch == 'P':
                        ohe[640+sq] = 1
                    elif ch=='p':
                        ohe[704+sq] = 1
                    else:
                         raise Exception(f'invalid FEN character {ch}')
                    sq += 1
        if sq != 64:
            raise Exception(f'invalid square count {sq}')

        return ohe, ptm


epd_parser = EPD()
my_epd = '1r4k1/p3bpp1/b2p1n2/3P4/2p1P3/P4PP1/1q2N3/R3KB1Q w - -'
ohe = epd_parser.toOHE(my_epd)
