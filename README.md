# Medleydb Playground
#### Final Project for [Music Information Retrieval at NYU](https://wp.nyu.edu/jpbello/teaching/mir/)

This project implements an end-to-end system for evaluating the performance of different feature representations and machine learning algorithms in melody extraction tasks of polyphonic music. It is heavily based and dependent on Medleydb and other work done by [Rachel Bittner](https://steinhardt.nyu.edu/marl/people/bittner) at NYU.

Code was written by [Willie Payne](williepayne.com) and Ana Elisa Mendez Mendez.

## Requirements
  * Python 3 (at least 3.4)
  * [Medleydb](http://medleydb.weebly.com) - Most dependencies are covered when following the directions to [install Medleydb](https://github.com/marl/medleydb)
  * [mir_eval](http://craffel.github.io/mir_eval/)

## Basic Usage
In order to run, navigate to the directory in which this code is stored and type the following in your terminal:
```
$ python3 playground.py
```
You will then be prompted to select three options:
```
Please choose a(n) evaluation mode:
0: melody
1: voicing
2: all
Your (integer) choice: 0
Please choose a(n) feature mode:
0: cube_root
1: stft
Your (integer) choice: 0
Please choose a(n) split mode:
0: validate
1: test
2: quick
Your (integer) choice: 2
```
Finally, let the program run! (Depending on your choices it may take a good amount of time...)
```
You chose to evaluate melody training with cube_root using quick data
Here we go!
Splitting Train and Test Sets..........Done
Extracting Training Features..........
Normalized MusicDelta_Reggae with 3009 feature vectors
Normalized MusicDelta_Rockabilly with 4471 feature vectors
Extracting Training Features..........Done
Removing Unvoiced Frames From Train..........Done
Training  Model..........
features array has shape:  (4232, 513)
labels array has shape:  (4232,)
Training  Model..........Done
Extracting Test Features..........
Normalized MusicDelta_Shadows with 5770 feature vectors
Extracting Test Features..........Done
Removing Unvoiced Frames From Test..........Done
Making Predictions..........Done
Exporting Predictions..........
Done: Exported file as predict_melody_cube_root_quick.json
Evaluating Results..........
guesses array has shape:  (4181,)
labels array has shape:  (4181,)
Evaluating pitch...
Evaluating chroma...
raw_chroma : 0.383161922985
raw_pitch : 0.260703181057
Evaluating Results..........Done
```
