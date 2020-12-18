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
So far, we do not use any existing code, but we are using the example project from the tutorial as basis for our project structure.

## Project State

### Planning State:
We have started with the downloading and processing of our data.

#### Done: 

##### Twitter API
- find Python library to work with Twitter API &rightarrow; [Tweepy](https://www.tweepy.org/ "Tweepy home")
- create credentials on Twitter developer portal to use the API
- become familiar to retrieve tweets by using library
- program python modul to receive data from the API

#### In Progress:

##### Collect labeled data sets
- find existing labeled data sets 
- check if data is useful
- how can different data sets combined
- program modul to read data sets (csv -files)
- split data into training / test set

We have so far checked the data set from our proposal, however since we have noticed (see diagram in Data Analysis Section) 
that a lot of tweets are missing we have decided to look for additional data sets and so far have found two more that could potentially be useful (see Data Analysis Section for further details on these).
For now we will be using the labels hateful and not-hateful (binary classification) and we plan to divide it into sub-categories if we find enough data. Thus, we will decide which labels could be summarized together, for more details see Data Analysis.


### Future Planning

#### Timeline:

![Timeline](/src/data/timeline.png)

#### Collect labeled data sets (12/27/2020) 
- find existing labeled data sets
- check if data is useful
- how can different data sets combined
- program modul to read data sets (csv -files)
- split data into training / test set


#### Preprocessing (01/04/2021) 
- program Python module to handle Preprocessing of tweets
- non standard lexical tokens have to filtered out (e.g. emoticons, hastags etc.)
  - tweets often includes urls like "http://t.co/cOU2WQ5L4q" and links with @name to other users, which should be removed, to make the data simpler
- remove duplicate tweets and retweets
- remove standard stopwords (english)
- splitting into tokens
- convert all tokens to lower case
- convert data with TF-IDF to make it ready to use for ML-algorithms

#### Train Classifier (01/18/2021) 
- train different classifier based on the train set (75% of the data set)
  - classifier:  
       - Support-Vector-Machine (SVM)
       - Decision Tree Classifier / Random- Forest Classifier 
       - Logistic Regression 
       - (Optional: Long Short Term Memory)
                 
#### Test Classifier (01/25/2021) 
- evaluate the different classifiers with the test set
- run m-fold cross-validation to determine the classifier with the smallest error
  - in addition to cross validation we want to use the F1-Score
  
#### Selection of meaninful tweets (02/01/2021) 
- select tweets from the twitter API which are:
  - english speech
  - located in United States (USA)
  - which were released in a specific time periode (e.g. time of US election)
  - tweets from a representable amount of people
    - people from different states of the USA
    - find threshold for which amount is representable

#### Analyze Data (Tweets) (02/08/2021) 
- execute preprocessing on selected twitter data
- use evaluated classifier to predict label of incoming tweets

#### Representation of data (02/15/2021) 
- plot data in an appropiate way

#### Presentation (02/20/2021) 

#### Project Report (03/10/2021) 


### High-level Architecture Description

#### Pipeline

![Pipeline](/src/data/pipeline.PNG)

#### Architecture
The basic structure of our project is planned as following: 
We plan to have one main file that is responsible for controlling any actions, all function can only be called from there.
Then we plan to have several sub packages divided by functionalities.
The first package is the package src.utils, which is meant to be responsible for the following functionalities:
+ functions for downloading and structuring the data from Twitter (including type of Tweet, no hate speech, racism and/or sexism and location of Tweet if available) 
+ functions for processing the data (for details of the planned steps of our processing pipeline refer to the section preprocessing)

The second package is the package src.data, which is meant to contain both IDs of the Tweets that build the basis of our training data and the processed data in form of a csv file. Furthermore, it should contain the data sets with directly the texts of the Tweets instead of the IDs as csv files. 

The third package is the package src.classifiers (will be added later, when implementing the classifiers), which is meant to be responsible for the following functionalities:
+ Support-Vector-Machine (SVM)  
+ Decision Tree Classifier / Random- Forest Classifier 
+ Logistic Regression 

Tests for all functionalities will be provided in the separate package tests.


### Experiments (so far)

See Sub Section Tweet Availability in Section Data Analysis.

## Data Analysis

### Data Source

As already mentioned in our proposal we are using an existing document collection, that can be found here:  
<https://github.com/zeerakw/hatespeech>   
This data set contains the IDs for Tweets and annotation for each tweet, whether they contain hate speech or not.

The data set (training/test set) and the data we want to investigate are retrieved from the Twitter API using the Tweepy library.


In additon we've found two other labeled data sets:

https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data
which includes labeled data for hate speech, offensive language and neither.

![Preview of second data set](/src/data/hatespeech_text_head.png)

The different labels used in detail are 'abusive', 'hateful', 'normal' and 'spam'.


The other data set we found is https://github.com/jaeyk/intersectional-bias-in-ml
which includes abusive language and hate speech.

![Preview of third data set](/src/data/labeled_data_head.png)

The class labels are those chosen by the majority of users who annotated the Tweets. 0 means 'hate speech', 1 means 'offensive language' and 2 means 'neither'.

Advantage of these two data sets is that they're including the tweets as raw text instead of ID's like in the first set. As already mentioned above, we have to decide which labels we are going to choose for our classifier, i.e. which labels can be summarized under 'hateful' or 'hate speech'.


### Preprocessing:

(0) Selection: the data we want to investigate have to be english speech tweets from the United States, which were released in a period of time (e.g. from may to august 2020). Moreover the tweets should be from a representive amount of people. <br>

(1) non standard lexical tokens have to filtered out (e.g. numbers, emoticons, hashtags..) <br>
(2) remove duplicate tweets and retweets <br>
(3) remove standard stopwords <br>
(4) splitting into tokens <br>
(5) convert all tokens to lower case <br>
(6) perform stemming    
(7) convert data with TF-IDF (uni- and bigrams allowed) to make it ready to use for machine learning algorithm <br>


### Basic Statistics:

 Data set <https://github.com/zeerakw/hatespeech>:
 
 The data set contains 16.907 annotated IDs, with the following availability distribution:  
 - 7.367/11.559 of normal content 
 - 11/1.970 of racism content 
 - 2.754/3.378 of sexist content 
 
 Location availability:
 - 122/11.559 of normal content
 - 1/1.970 of racism content 
 - 38/3.378 of sexist content 
 
  ![Tweets availability](/src/data/tweets_availability_per_label_zeerak.png)
 
 
 Data set https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data:
  
 The data set contains 24.783 annotated tweets, with the following distribution:  
 - 4.163 of normal content 
 - 19.190 of offensive language content 
 - 1.430 of hate speech content 
 
  ![Tweets availability](/src/data/tweets_per_label_tdavidson.png)
   
 Data set https://github.com/jaeyk/intersectional-bias-in-ml:
  
 The data set contains 99.996 annotated tweets, with the following distribution:  
 - 53.851 of normal content 
 - 27.150 of abusive content 
 - 14.030 of spam content  
 - 4.965 of hateful content
  
 ![Tweets availability](/src/data/tweets_per_label_jaeyk.png)
   
 
 ### Examples
 
 Data set <https://github.com/zeerakw/hatespeech>:
 
 Normal example: "If something is so hard to do on the BBQ then why why why do it?? #MKR #hungrycampers" <br>
 Racism example: "These girls are the equivalent of the irritating Asian girls a couple years ago. Well done, 7. #MKR" <br>
 Sexism example: "Trying to find something pretty about these blonde idiots.#MKR" <br> <br> 
 
 Data set https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data:
 Normal example: "Pit Bulls Photographed As Lovely Fairy Tale Creatures" <br>
 Offensive language example: "As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out..." <br>
 Hate speech example: ""@white_thunduh alsarabsss" hes a beaner smh you can tell hes a mexican" <br> <br>

 Data set https://github.com/jaeyk/intersectional-bias-in-ml:
 Normal example: "Topped the group in TGP Disc Jam Season 2! Onto the Semi-Finals!" <br>
 Abusive example: "RT @Papapishu: Man it would fucking rule if we had a party that was against perpetual warfare." <br>
 Spam example: "4X DIY Birds Stencil Cutting Carbon Scrapbooking Card Diary Stamping Template" <br>
 Hateful example: "I'm over the fucking moon we've cleared up the definition of an act of war. Now, about that slap on the wrist we just gave Syria." <br> <br>
 

 

 

