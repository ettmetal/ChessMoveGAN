import chess
import chess.pgn
import tensorflow as tf
import numpy as np
from datetime import date
from helpers.conversion import AZ_MOVE_COUNT, board_to_matrix, az_index_to_move, move_to_az_index
from helpers.iteration import iter_until_except
from models.ccgan import load_models


def generate_headers_for(game):
    # Include both FICS style and PGN standard headers for player type
    return {
        "Event": "?",
        "Site": "?",
        "Date": date.today().strftime("%Y.%m.%d"),
        "Round": "-",
        "White": "ChessMoveGAN",
        "Black": "ChessMoveGAN",
        "Result": game.end().board().result(),
        "WhiteIsComp": "Yes",
        "BlackIsComp": "Yes",
        "WhiteType": "program",
        "BlackType": "program",
        "PlyCount": "%s" % (game.end().board().fullmove_number + 1 if game.end().board().turn == False else 0)
    }


def generate_game(board):
    # make game with headers
    game = chess.pgn.Game.from_board(board)
    for header, value in generate_headers_for(game).items():
        game.headers[header] = value
    return game

def generate_boards(generator, sess):
    # Set up a board
    board = chess.Board()

    while not board.is_game_over(claim_draw=True):
        in_board = [board_to_matrix(board)]
        in_noise = sess.run(tf.random.normal((1,1)))
        probabilities = generator.predict([in_board, in_noise], steps=1)[0]

        board.push(az_index_to_move(highest_probability_valid_move(board, probabilities)))

    return board


def highest_probability_valid_move(board, probabilities):
    legal_az_move_indicies = [move_to_az_index(move) for move in board.legal_moves]
    invalid_indicies = [i for i in range(AZ_MOVE_COUNT) if i not in legal_az_move_indicies]
    for invalid in invalid_indicies:
        probabilities[invalid] = 0
    return np.argmax(probabilities)


def main():
    tf.logging.set_verbosity(tf.logging.ERROR) # ignore some irritating, but benign, tensorflow warnings

    games_to_generate = 3
    pgn_out_path = ".pgn"
    # Get the generator
    model_dir = ""
    with tf.Session() as sess:
        generator, _, _ = load_models(model_dir)

        # Generate games and add to PGN file
        with open(pgn_out_path, 'w+') as pgn_out_file:
            for game_count in range(games_to_generate):
                print("Generating game %s of %s" % (game_count, games_to_generate))
                game = generate_game(generate_boards(generator, sess))
                print(game, file=pgn_out_file, end="\n\n")


if __name__ == "__main__":
    main()
