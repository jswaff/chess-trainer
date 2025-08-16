# utility script to merge win/draw/loss counts for each position
# input lines should be sorted and of format fen,score,wins,draws,losses
# Examples:
#      rnbqkb1r/ppp1pp1p/6p1/3n4/3PP3/2N5/PP3PPP/R1BQKBNR b KQkq e3,-59,1,0,0
#      rnbqkb1r/ppp1pp1p/6p1/3n4/3PP3/2N5/PP3PPP/R1BQKBNR b KQkq e3,-59,1,2,0
#      rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq -,-8,0,1,1
#
# The output lines will be the same format: fen,score,wins,draws,losses
# Examples:
#      rnbqkb1r/ppp1pp1p/6p1/3n4/3PP3/2N5/PP3PPP/R1BQKBNR b KQkq e3,-59,2,2,0
#      rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq -,-8,0,1,1
#
# Note lines with duplicate FENs have been consolidated

ifile = "data/positions-sorted.csv"
ofile = "data/positions-merged.csv"

with open(ifile, "r") as infile, open(ofile, "w") as outfile:
    current_line = infile.readline()
    wins = 0
    draws = 0
    losses = 0

    while current_line:
        current_line = current_line.strip()
        [fen, score, curr_wins, curr_draws, curr_losses] = current_line.split(',',maxsplit=4)
        wins += int(curr_wins)
        draws += int(curr_draws)
        losses += int(curr_losses)

        next_line = infile.readline().strip()
        next_is_same = False
        if next_line:
            [next_fen, _, _, _, _] = next_line.split(',',maxsplit=4)
            next_is_same = fen == next_fen

        if not next_is_same:
            outline = fen + ',' + score + ',' + str(wins) + ',' + str(draws) + ',' + str(losses) + '\n'
            outfile.write(outline)
            wins = 0
            draws = 0
            losses = 0

        current_line = next_line
