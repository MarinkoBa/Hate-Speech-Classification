# Hate-Speech-Classification

### Teammembers and Mail addresses:
- Team Member: Balikic Marinko,
kf252@stud.uni-heidelberg.de
- Team Member: Bopp Sarah,
Bopp@stud.uni-heidelberg.de
- Team Member: Gebhard Fabio,
fk269@stud.uni-heidelberg.de
- Team Member: Tratz-Weinmann Daniela,
Tratz-Weinmann@stud.uni-heidelberg.de

### Existing code fragments:
So far, we do not use any existing code.

## Project State

### Planning State:
We have started with the downloading and processing of our data.

### Future Planning

Hier noch timeline einfügen und sinnvolle Fristen

#### Collect labeled data sets
- find existing labeled data sets
- check if data is useful
- how can different data sets combined
- program modul to read data sets (csv -files)
- split data into training / test set

#### Twitter API
- find Python library to work with Tiwtter API
- create credentials on Twitter developer portal to use the API
- become familiar to retrieve tweets by using library
- program python modul to receive data from the API

#### Preprocessing
- program Python module to handle Preprocessing of tweets
- non standard lexical tokens have to filtered out (e.g. emoticons, hastags etc.)
  - tweets often includes urls like "http://t.co/cOU2WQ5L4q" and links with @name to other users, which should be remeoved, to make the data simpler
- remove duplicate tweets and retweets
- remove standard stopwords (english)
- splitting into tokens
- convert all tokens to lower case
- convert data with TF-IDF to make it ready to use for ML-algorithms

#### Train Classifier
- train different classifier based on the train set (75% of the data set)
  - classifier:  
       - Support-Vector-Machine (SVM)
       - Decision Tree Classifier / Random- Forest Classifier 
       - Logisitc Regression 
       - Long Short Term Memory 
                 
#### Test Classifier
- evaluate the different classifiers with the test set
- run m-fold-cross validation to determine the classifier with the smallest error
  - in addition to cross validation we want to use the F1-Score
  
#### Selection of meaninful tweets
- select tweets from the twitter API which are:
  - english speech
  - located in United States (USA)
  - which were released in a specific time periode (e.g. time of US election)
  - tweets from a representable amount of people
    - people from different states of the USA
    - find threshold for which amount is representable

#### Analyze Data (Tweets)
- execute preprocessing on selected twitter data
- use evaluated classifier to predict label of incoming tweets

#### Representation of data
- plot data in an appropiate way


### High-level Architecture Description

Suggestion:
The basic structure of our project is planned as following: 
We plan to have one main file that is responsible for controlling any actions, all function can only be called from there.
Then we plan to have several sub packages divided by functionalities.
The first package is the package src.utils, which is meant to be responsible for the following functionalities:
+ functions for downloading and structuring the data from Twitter (including type of Tweet, no hate speech, racism and/or sexism and location of Tweet if available) 
+ functions for processing the data ( for details of the planned steps of our processing pipeline refer to [this section](###-preprocessing:))
The second package is the package src.data, which is meant to contain both Ids of the Tweets that build the basis of our training data and the processed data in form of a csv file 

The third package is the package src.classifiers, which is meant to be responsible for the following functionalities:
+ function for classification using logistic regression
+ other method: vielleicht tree?
+ noch eine Methode?

Tests for all functionalities will be provided in the separate package tests.


### Experiments

vielleicht datenaufteilung?

## Data Analysis

### Data Source

As already mentioned in our proposal we are using an exsiting document collection, that can be found here:  
<https://github.com/zeerakw/hatespeech>   
This data set contains the IDs for Tweets and annotation for each tweet, whether they contain hate speech or not.

The data set (training/test set) and the data we want to investigate are retrieved from the Twitter API using the Tweepy library.

### Preprocessing:

(0) Selection: the data we want to investigate have to be english speech tweets from the United States, which were released in a period of time (e.g. from may to august 2020). Moreover the tweets should be from a representive amount of people. <br>

(1) non standard lexical tokens have to filtered out (e.g. numbers, emoticons, hashtags..) <br>
(2) remove duplicate tweets and retweets <br>
(3) remove standard stopwords <br>
(4) splitting into tokens <br>
(5) convert all tokens to lower case <br>
(6) convert data with TF-IDF (uni- and bigrams allowed) to make it ready to use for machine learning algorithm <br>
<br>
Frage? Stemming scheint nicht notwendig zu sein!? <br>

### Basic Statistics:

 The data set contains 16.907 annotated IDs, with the following distribution:  
 - 1.970 of racism content 
 - 3.378 of sexist content 
 - 11.559 with none of them are 
 
 Location availability:
 - 1/1.970 of racism content 
 - 38/3.378 of sexist content 
 - 122/11.559 with none of them are 
 
 #### Tweets availability
 ![Tweets availability](/src/data/tweets_availability_per_label.png)
 
 
 ### Examples
 
 Racism example: "These girls are the equivalent of the irritating Asian girls a couple years ago. Well done, 7. #MKR" <br>
 Sexism example: "Trying to find something pretty about these blonde idiots.#MKR" <br>
 None example: "If something is so hard to do on the BBQ then why why why do it?? #MKR #hungrycampers" <br> <br>
 
 Mindestens ein example sample für unser data, also vielleicht ein paar Tweets (pro Kategorie einer)?
 
 ### was muss bis Freitag gemacht werden:
 + requirements.txt, falls wir schon irgendwelche libraries benutzen
 + möglichst schon code für download/preprocessing
 + Zeitplan
 + Projektaufbau
 + Betreuer zugriff auf repo geben | erledigt
 
 

