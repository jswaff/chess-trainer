# preprocess.py
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

import argparse


def reconsolidate_and_clamp(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        current_line = infile.readline()
        wins = 0
        draws = 0
        losses = 0

        while current_line:
            current_line = current_line.strip()
            if not current_line:
                current_line = infile.readline()
                continue

            fields = current_line.split(",", maxsplit=4)
            if len(fields) != 5:
                raise ValueError(f"Invalid input line: {current_line}")

            fen, score, curr_wins, curr_draws, curr_losses = fields
            wins += int(curr_wins)
            draws += int(curr_draws)
            losses += int(curr_losses)

            next_line = infile.readline()
            while next_line and not next_line.strip():
                next_line = infile.readline()

            next_is_same = False
            if next_line:
                next_fields = next_line.strip().split(",", maxsplit=4)
                if len(next_fields) != 5:
                    raise ValueError(f"Invalid input line: {next_line.strip()}")
                next_fen = next_fields[0]
                next_is_same = fen == next_fen

            if not next_is_same:
                iscore = int(score)
                if abs(iscore) < 32000:
                    if iscore < -1500:
                        iscore = -1500
                    elif iscore > 1500:
                        iscore = 1500

                    outline = ",".join(
                        [fen, str(iscore), str(wins), str(draws), str(losses)]
                    ) + "\n"
                    outfile.write(outline)

                wins = 0
                draws = 0
                losses = 0

            current_line = next_line


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a sorted chess position CSV, merge duplicate positions, "
            "clamp scores, and write the final output CSV."
        )
    )
    parser.add_argument("ifile", help="Path to the input CSV file")
    parser.add_argument("ofile", help="Path to the output CSV file")
    args = parser.parse_args()

    reconsolidate_and_clamp(args.ifile, args.ofile)


if __name__ == "__main__":
    main()
