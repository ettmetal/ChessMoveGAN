import chess
import chess.svg
from helpers.io_helpers import ensure_dir_exists
from helpers.pgn_helpers import pgn_iterate


def main():
    # Input
    input_path = "generated.pgn"
    # Output
    output_directory = input_path.replace(".pgn", "") + "/"
    ensure_dir_exists(output_directory)

    with open(input_path, 'r') as games_file:
        game_counter = 0
        for game in pgn_iterate(games_file):
            print("Writing svgs for game %s" % game_counter)
            # Subdirectory for each game
            game_directory = (output_directory + "%s/") % game_counter
            ensure_dir_exists(game_directory)

            # Get starting board and write it out
            board = chess.Board()
            board_counter = 0
            output_path = game_directory + "%s.svg"
            with open(output_path % board_counter, 'w+') as output:
                output.write(chess.svg.board(board))

            # Write the board after each move
            for step in game.mainline():
                board_counter += 1
                with open(output_path % board_counter, 'w+') as output:
                    output.write(chess.svg.board(step.board, arrows=[(step.move.from_square, step.move.to_square)], lastmove=step.move))

        game_counter += 1

if __name__ == "__main__":
    main()
