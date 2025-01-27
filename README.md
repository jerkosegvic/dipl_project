# Training BERT as a Retriver part of RAG pipeline

### Zip file with all datasets
- https://drive.google.com/file/d/1tZFHkD0ohgFZjlp-CYNW4GYkA3Dvmx1P/view?usp=drive_link

### Multichoice datasets
- SuperGLue/MultiRC (https://cogcomp.seas.upenn.edu/multirc/), Multi choice, multiple correct answer

### Usage
- install requierments witw `pip install -r requierments.txt`
- `train_multirc_retriver.py` is a script for training retriver with shared encoder 
- `train_multirc_dual_retriver.py` is a script for training retriver with two encoders, one for questions and one for sentences
- before running any of those, make sure to set paths and hyperpaameters correctly
- run the script
- after training the model(s) configure paths in `plotting.py` script
- run `plotting.py` to get all metrics
