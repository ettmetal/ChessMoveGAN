import chess.pgn

# PGN helper functions

def count_games_in_pgn_file(handle):
    counter = 0
    while chess.pgn.skip_game(handle):
        counter += 1
    return counter


# PGN iteration
def pgn_iterate(handle):
    game = chess.pgn.read_game(handle)
    while game is not None:
        yield game
        game = chess.pgn.read_game(handle)


def pgn_iterate_headers(handle):
    headers = chess.pgn.read_headers(handle)
    while headers is not None:
        yield headers
        headers = chess.pgn.read_headers(handle)
