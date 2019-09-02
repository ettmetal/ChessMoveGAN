from helpers.io_helpers import ensure_dir_exists
from helpers.pgn_helpers import count_games_in_pgn_file, pgn_iterate
import msgpack

# Set these variables to ensure correct in/out paths
pgn_path = "games.pgn"
out_directory = "preprocessed_games/%s"
board_states_path = out_directory % "boards__standard_2018_human_morethan10.mpk"
moves_path = out_directory % "moves__standard_2018_human_morethan10.mpk"

min_length = 10

log_freq = 1 # How frequently status is logged to the console


# Reads in a .pgn file, and converts each game into sequences of board states (data/x) and moves (labels/y).
# Games are filtered to only those where all players were human and more than 10 half-moves were played.
# Board state and move sequences are written, preserving order, to seperate files as messagepack'ed python lists.
def main():
    # Logging counters
    game_counter = 0
    skipped_games = 0
    skipped_computer = 0
    skipped_ply = 0
    games_since_last_log = 0
    total_moves = 0

    ensure_dir_exists(out_directory.replace("%s", ""))
    with open(pgn_path, 'r') as pgn_file, open(board_states_path, 'ab+') as boards_out_file, open(moves_path, 'ab+') as moves_out_file:
        # Count games and reset seek position
        input_game_count = count_games_in_pgn_file(pgn_file)
        pgn_file.seek(0)
        print("Found %s games to process" % input_game_count)

        for game in pgn_iterate(pgn_file):
            games_since_last_log += 1

            # Don't process games which meet filter criteria
            ply = int(game.headers["PlyCount"]) # Get number of half-moves in the current game
            # FICS uses these headers for player type instead of the standard
            has_computer_player = game.headers.get("WhiteIsComp") == "Yes" or game.headers.get("BlackIsComp") == "Yes"
            if has_computer_player:
                skipped_games += 1
                skipped_computer += 1
            elif ply <= min_length:
                skipped_games += 1
                skipped_ply += 1
            else: # Convert the game and pack it
                processed_boards = game_to_board_list(game)
                processed_moves = game_to_az_indices(game)

                total_moves += len(processed_boards)
                msgpack.pack(processed_boards, boards_out_file)
                msgpack.pack(processed_moves, moves_out_file)

            # Log current status to console every log_freq games
            if games_since_last_log >= log_freq:
                print("Processing game %s of %s" % (game_counter + 1, input_game_count))
                print("\tSkipped %s games with computer players\n\tSkipped %s games with %s or fewer half-moves." % (skipped_computer, skipped_ply, str(min_length - 1)))

                # Reset reporting counters
                games_since_last_log = 0
                skipped_computer = 0
                skipped_ply = 0

            game_counter += 1
    print("Wrote %s moves in %s games of %s input games" % (total_moves, game_counter - skipped_games, input_game_count))


if __name__ == "__main__":
    main()
