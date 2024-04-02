# Neo Controller

## Importing and Running
Python versions 3.10 and 3.11 are supported.
Copy the corresponding compiled Python extension module files into a place of your choosing:

`neo_controller.cp310-win_amd64.pyd` for 3.10

`neo_controller.cp311-win_amd64.pyd` for 3.11

To import the extension module, do `from neo_controller import NeoController`

IMPORTANT: Make sure that neo_controller.py is not also in the directory you are importing from, otherwise you might be importing the interpreted Python script instead of the compiled extension module. The compiled version executes ~6 times faster than the interpreted version, and performs better in-game as a result.

## Additional Info
neo_controller.py is the preprocessed version of neo_controller.py from the root directory, created using competition_neo_preprocessor.py. Debug code and unnecessary stuff have been stripped out, and this slightly improves performance as dead code and unnecessary branches aren't being run.
