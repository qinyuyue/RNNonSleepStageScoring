## CSE 6250: Big Data for Healthcare Project: Automatic Sleep Staging
### User Guide - Team 28 Sleep Data

1. Description  
	We created a recurrent neural network to automate the sleep scoring process.

2. Setup  
	Inside the top directory, run:
	~~~
	conda env create -f environment.yml
	~~~
	Activate the environment:
	~~~
	source (conda) activate sleep
	~~~
	Install PyTorch according to your OS (https://pytorch.org/)

3.  To test the trained model on the 6 sample patients:
    ~~~
    python src/test_best_model.py
    ~~~
   
4.  To train a new model, you must first update the constants.py file to point to the right directories. Then run:
    ~~~
	python src/list.py  //this splits the processed data into train and test sets
	python src/main.py  //this trains the model, outputs the results into the outputs dir, and the graphs into the graphs dir
	~~~

Our subset of sample processed data is in the /submission_test directory.  
Our saved model is in the /submission_test directory.
# RNNonSleepStageScoring
