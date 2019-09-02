import msgpack
from random import sample

# Number of games after filtering by preprocessing
games_count = 557049
# In/out paths
all_boards_path = "preprocessed_games/boards__standard_2018_human_morethan10.mpk"
all_moves_path = "preprocessed_games/moves__standard_2018_human_morethan10.mpk"
boards_output_path_like = "preprocessed_games/%s__boards__standard_2018_human_morethan10.mpk"
moves_output_path_like = "preprocessed_games/%s__moves__standard_2018_human_morethan10.mpk"

# Numbers of games to sample, with one pair of files output for each size
sample_sizes = [10, 100]

def main():
    # Select games by index for each sample
    indicies_by_sample = {sample: sorted(sample(range(games_count), sample)) for sample in sample_sizes}
    print("Writing %s samples:" % len(indicies_by_sample), *sample_sizes, sep="\n  ")

    from contextlib import ExitStack
    with open(all_boards_path, 'rb') as game_input, open(all_moves_path, 'rb') as label_input, ExitStack() as stack:
        game_unpacker = msgpack.Unpacker(game_input)
        label_unpacker = msgpack.Unpacker(label_input)

        # A pair of output files for each sample
        outfiles = {sample: {
                    "moves_file": stack.enter_context(open(moves_output_path_like % sample, 'ab+')),
                    "boards_file": stack.enter_context(open(boards_output_path_like % sample, 'ab+')),
                } for sample in sample_sizes }

        print("Have %s out file pairs" % len(outfiles))

        # For counting total moves in each sample
        counts = {sample: 0 for sample in sample_sizes}

        # Iterate all games, appending that game to all sample sizes it has been selected for
        for index in range(games_count):
            these_boards = None
            these_moves = None
            for sample in sample_sizes:
                if index in indicies_by_sample[sample]:
                    # Get boards & moves in current game (only once per game)
                    these_moves = these_moves or label_unpacker.unpack()
                    these_boards = these_boards or game_unpacker.unpack()
                    
                    moves_in_game = len(these_boards)
                    print("Packing game %s for sample %s. Contains %s moves" % (index, sample, moves_in_game))

                    # Append all boards & moves in the game to appropriate out files for the sample
                    for i in range(len(these_boards)):
                        msgpack.pack(these_boards[i], outfiles[sample]['boards_file'])
                        msgpack.pack(these_moves[i], outfiles[sample]['moves_file'])

                    counts[sample] += moves_in_game

                    # Ensure each game is included only once
                    indicies_by_sample[sample].remove(index)
            # Skip game if it appears in no samples
            if these_boards is None:
                label_unpacker.skip()
                game_unpacker.skip()

            # Count number of samples with no remaining games to sample to allow early exit
            empties = 0
            for sample in sample_sizes:
                empties += 1 if len(indicies_by_sample[sample]) == 0 else 0
            if empties == len (sample_sizes):
                break

    for sample in sample_sizes:
        print("Split %s contains %s moves" % (sample, counts[sample]))

if __name__ == "__main__":
    main()
