# Udacity Disaster Response Pipeline #


## Description ##

This project was created to satisfy the project-2 requirements of the [Udactiy Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) course.

The goal of the project is to train a machine learning pipeline accept a message sent durring a disaster (on text, twitter, facebook, etc...) and classify this message based on what an aid worker could do about it (e.g. search and rescue, fire, water, aid, etc...). To accomplish this goal, a machine learning model was developed and trained using labeled message data from [Figure Eight](https://www.figure-eight.com/). The other section of this project is a deployable web app that allows a user to enter a message to categorize, and displays the result.


## Requirements ##

For the ETL Pipeline (in process_data.py) this project used *numpy*, *pandas*, and *sqlalchemy* to load, clean, and save the data.

For the ML Pipeline (in train_classifier.py) this project used *pandas* and *sqlalchemy* to load the data, *nltk* and *sklearn* to build and train the model, and *pickle* to save the model.

For the Web App (in the app folder) this project used *json*, *plotly*, and *flask* for displays and the web page, and *nltk*, *sklearn*, and *sqlalchemy* for the interactive message categorizer.


## Usage ##

- Run the following commands in the project's root directory to set up your database and model.

	- To run ETL pipeline that cleans data and stores in database
    	`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
	- To run ML pipeline that trains classifier and saves
    	`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- Run the following command in the app's directory to run your web app.
    `python run.py`

- Go to http://0.0.0.0:3001/

## Included files ##

- app
	- templates
		- master.html  # main page of web app
		- go.html  # classification result page of web app
	- run.py  # Flask file that runs app

- data
	- disaster_categories.csv  # data to process 
	- disaster_messages.csv  # data to process
	- process_data.py  # script to run an ETL pipeline on raw data
	- InsertDatabaseName.db   # database to save clean data to

- models
	- train_classifier.py  # script to train a predictive model
	- classifier.pkl  # saved model 

- README.md