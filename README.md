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
+ Preprocessing
+ Implementing methods of classification
+ Train methods
+ Test/ evaluate
+ Interpretation/plotting/analysing of test results

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

 The data set contains 16.914 annotated IDs, with the following distribution:  
 - 1.972 of racism content 
 - 3.383 of sexist content 
 - 11.559 with none of them are 
 
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
 
 

