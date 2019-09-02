# ChessMoveGAN

An attempt at training a Generative Adversarial Network to generate chess moves, conditioned on board state.

## Use

### Data

Input games are assumed provided in a single .pgn file.

Games are pre-processed by running the preprocessing.py script, ensuring that in/out file paths are set.

Then, the number(s) of games required are sampled using sample_games.py, again with in/out path variables set, also with `games_count` set to the number of games written by preprocessing.py and `sample_sizes` set to a list of sample sizes to create. Each of these samples will have its own pair of output files. This script also splits games such that the output is two lists: one of boards and the other of matching moves, with no retained relationship to the game they were part of.

#### Source

Games were acquired from the [FICS database][1]. These were all standard games of any rating for the whole year of 2018.

### Training

To train, run chessmovegan.py. This file requires `boards_path` and `moves_path` to be set to files from the same sample. Optionally, `run_name` can be set so that stored ouput will be identifiable.

### Generation

Generating game logs is handled by the script generate_logs.py. To change the number of games generated, change `games_to_generate`. Games will be written to the file at `pgn_out_path`, the file name of which should end .pgn.

#### Generating SVGs

Included is a script to parse games from a .pgn file and render each game therein to a series of svgs. That script is visualise_games_as_svg.py. Set the `input_path` variable to select the input .pgn file.

### Debugging

The script chessmovegan_instrumented.py uses training which includes time profiling for each step in a batch plus RAM usage each epoch. It has the additional dependancy (blah). In all other ways, it is identical to chessmovegan.py

## Dependancies

- [Python] v >= 3.6.5
- [Tensorflow] v >= 1.13.1
- [msgpack] v >= 0.6.1
- [python-chess] v >= 0.26.0

[1]: https://www.ficsgames.org/
[Python]: https://www.python.org/
[Tensorflow]: https://www.tensorflow.org/
[msgpack]: https://github.com/msgpack/msgpack-python
[python-chess]: https://github.com/niklasf/python-chess
