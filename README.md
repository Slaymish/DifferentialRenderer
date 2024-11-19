project structure:
- code/
    - models/
        - Contains the model files for the project. Each model should be in its own file.
    - tests/
        - Contains the test files for the project. Each test should be in its own file.
    - utils/
        - data_utils.py
        - image_utils.py
        - model_utils.py
        - train_utils.py
- data/
    - images/
    - text/
    - malwares/
- model_checkpoints/
    - Contains the saved models for the project.
- Obsidian/
  - Contains the Obsidian vault for the project, which includes all the notes, ideas, and resources for the project.
- README.md
- requirements.txt


notes:
- each model file should have a main() function that can be called from the command line, which will train the model and save it to a specified directory

## Setup

1. Create a virtual environment and activate it.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the main function of the model file to train the model and save it to the model_checkpoints directory.