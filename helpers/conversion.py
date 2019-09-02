import chess
import math
from .coordinates import *

# -- Magic values ------------------------------------------------------------------------------------------------------
AZ_MOVE_COUNT = 4672

AZ_INDEX_NONE = -1

AZ_SHAPE = (8, 8, 73)

UNDERPROMOTIONS_SHAPE = (3, 3)

PROMOTION_PIECE_MODIFIER = 2

QUEEN_MOVE_SHAPE = (8, 7)


# -- Whole-game conversion ---------------------------------------------------------------------------------------------
# Transform a python-chess game (mainline) into AlphaZero-like moves (indices)
def game_to_az_indices(game):
    indices = [move_to_az_index(move) for move in game.mainline_moves()]
    indices.append(AZ_INDEX_NONE)
    return indices


# Transform a python-chess game (mainline) into AlphaZero-like moves (one-hot)
def game_to_az(game):
    return [move_to_az(move) for move in game.mainline_moves()]


# Transform a python-chess game (mainline) into a sequence of board states
# Boards are 8x8x12, 8x8 squares with a one-hot piece
def game_to_board_list(game):
    boards = []
    board = game.board()
    boards.append(board_to_matrix(board))
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board_to_matrix(board))
    return boards


# Transform a python-chess board into an 8x8x12 board of one-hot piece types
def board_to_matrix(board):
    pieces = board.piece_map()
    board_matrix = [[] for _unused in range(8)]
    for rank in range(8):
        for file in range(8):
            piece = pieces.get(chess.square(file, rank))
            board_matrix[rank].append(one_hot_piece_vector(piece))
    return board_matrix


# -- Piece conversion --------------------------------------------------------------------------------------------------
# Converts a piece to nominal encoding, then converts to one-hot
def one_hot_piece_vector(piece):
    return one_hot_nominal_piece_vector(convert_piece_to_nominal(piece.piece_type if piece else 0, piece.color if piece else False))


# Converts a nominally-encoded piece to a one-hot encoding
def one_hot_nominal_piece_vector(piece_type):
    # 12 piece types
    one_hot = [0] * 12
    if piece_type == 0:
        return one_hot
    one_hot[piece_type - 1] = 1
    return one_hot


# Convert piece type and colour to a nominally-encoded piece
def convert_piece_to_nominal(piece_type, piece_color):
    return 0 if piece_type == 0 else piece_type + ((0 if piece_color else 1) * 6)


# -- AlphaZero encodings -----------------------------------------------------------------------------------------------
# -- To AlphaZero ------------------------------------------------------------------------------------------------------
# Convert move to AlphaZero index, then convert to one-hot
def move_to_az(move):
    return az_index_to_az(move_to_az_index(move))


# Convert a Move to an AlphaZero index
def move_to_az_index(move):
    if move is None or move.from_square is None or move.to_square is None:
        return AZ_INDEX_NONE  # Use a special case index for null moves
    departure_file = chess.square_file(move.from_square)
    departure_rank = chess.square_rank(move.from_square)
    if move.promotion is not None and move.promotion is not chess.QUEEN:
        move_offset = 64 + get_underpromotion_type(move)  # range: 64-72
    elif is_knight_move(move):
        move_offset = 56 + encode_knight_move(move)  # range: 56-63
    else:  # All other moves (including Queen promotions) are encoded as follows
        move_offset = encode_queen_move(move)  # range: 0-55
    return flatten_3d_coords((departure_file, departure_rank, move_offset), AZ_SHAPE)


# -- From AlphaZero ----------------------------------------------------------------------------------------------------
# Transform an AlphaZero-like move index into a python-chess move
def az_index_to_move(index):
    if index == -1:
        return chess.Move(None, None)
    move_coordinates = inflate_1d_coords_3d(index, AZ_SHAPE)
    from_file = move_coordinates[0]
    from_rank = move_coordinates[1]
    move_type = move_coordinates[2]
    promotion_piece = None

    if move_type >= 64:  # Treat under-promotions as 3 x 3 array of [type of file movement, piece promoted to]
        underpromotion_type = inflate_1d_coords_2d(move_type - 64, UNDERPROMOTIONS_SHAPE)  # Get under-promotion coords

        file_modifier = underpromotion_type[0] - 1  # shift from range 0,..2 to -1,..1
        promotion_piece = underpromotion_type[1] + PROMOTION_PIECE_MODIFIER  # Restore to python-chesspiece numbers

        to_file = from_file + file_modifier
        to_rank = 7 if from_rank == 6 else 0  # Under-promotions always move a rank ahead (player-relative)
    elif move_type >= 56:
        knight_delta = decode_knight_move_delta(move_type - 56)

        to_file = from_file + knight_delta[0]
        to_rank = from_rank + knight_delta[1]
    else:
        #its a queen move
        queen_delta = decode_queen_delta(move_type)
        to_file = from_file + queen_delta[0]
        to_rank = from_rank + queen_delta[1]

    return chess.Move(chess.square(from_file, from_rank), chess.square(to_file, to_rank), promotion_piece)


# Convert an AlphaZero-like move index to one-hot
def az_index_to_az(index):
    az = [0] * AZ_MOVE_COUNT
    valid_check = False
    try:
        valid = az.index(1)
    except ValueError:
        valid_check = True
    if not valid_check:
        raise ValueError('something went wrong and the blank array was not')
    if index == AZ_INDEX_NONE:
        return az
    if 0 <= index <= AZ_MOVE_COUNT:
        az[index] = 1
    return az


# Covert one-hot to an AlphaZero index
def az_to_az_index(az):
    try:
        return az.index(1)
    except ValueError:
        return AZ_INDEX_NONE


# -- AlphaZero helpers -------------------------------------------------------------------------------------------------
# -- Queen helpers -----------------------------------------------------------------------------------------------------
def encode_queen_move(move):
    # Only need 7 distances (1,2,3,4,5,6,7) when applied to queen-type moves (no non-moves so no 0)
    distance = chess.square_distance(move.from_square, move.to_square) - 1

    file_delta = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    rank_delta = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    heading = math.atan2(file_delta, rank_delta)
    direction = round((math.degrees(heading) % 360) / 45)  # Convert heading to cardinal direction: N=0, NE=1 etc...

    return flatten_2d_coords((direction, distance), QUEEN_MOVE_SHAPE)


def decode_queen_delta(queen_move):
    queen_coords = inflate_1d_coords_2d(queen_move, QUEEN_MOVE_SHAPE)
    queen_heading = math.radians(queen_coords[0] * 45)
    queen_distance = queen_coords[1] + 1
    delta = (round(math.sin(queen_heading)) * queen_distance, round(math.cos(queen_heading)) * queen_distance)
    return delta


# -- Knight helpers ----------------------------------------------------------------------------------------------------
def is_knight_move(move):
    file_delta = abs(chess.square_file(move.to_square) - chess.square_file(move.from_square))
    rank_delta = abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square))
    # Knight Moves have the unique (in chess) characteristic: abs(dx) != 0 && abs(dy) != 0 && dy != dx && dy == 0
    return file_delta != 0 and rank_delta != 0 and file_delta != rank_delta


def encode_knight_move(move):
    def encode_delta(delta):
        return 0 if delta >= 0 else 1
    file_delta = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    rank_delta = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    move_type = 0 if abs(rank_delta) > abs(file_delta) else 4  # Provides 8 unique encodings without 8 ifs
    return move_type + 2 * encode_delta(file_delta) + encode_delta(rank_delta)


def decode_knight_move_delta(knight_move):
    def decode_file(x):
        return 1 if x < 2 else -1

    def decode_rank(x):
        return 1 if x % 2 == 0 else -1

    if knight_move < 4:
        file_delta = decode_file(knight_move)
        rank_delta = 2 * decode_rank(knight_move)
    else:
        file_delta = 2 * decode_file(knight_move - 4)
        rank_delta = decode_rank(knight_move - 4)

    return (file_delta, rank_delta)


# -- Under-promotions --------------------------------------------------------------------------------------------------
# encode an underpromotion move as a flattened index into a 3x3 space
def get_underpromotion_type(move):
    file_delta = chess.square_file(move.to_square) - chess.square_file(move.from_square)  # Get heading
    promotion_destination = file_delta + 1  # Encoded as move left = 0, move forward = 1, move right = 2
    return flatten_2d_coords((promotion_destination, move.promotion - PROMOTION_PIECE_MODIFIER), UNDERPROMOTIONS_SHAPE)
