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
 We do not use any existing code, but we are using the example project from the tutorial as basis for our project structure and existing libraries thnat are listed in the requirements.txt.
 

## Project State

### TODO
- N-Grams (Daniela) Done
- Balance Tweets (Marinko) Done
- Location Tweets (Sarah) Done
- auf Karte markieren? (nur wenn Ergebnisse interessant sind bzw. gut darstellbar) (Marinko) Done
- Tests (Typen etc.) (Daniela) 
- Dateipfade anpassen + Instructions ergänzen / Anleitung (Fabio)
- Optionen testen: balanced als Grundlage: preprocessing 2x (für eine Methode entscheiden) (Fabio), unigram, bigram, method,f1-score (tfidf oder count)(Marinko) Done
- Kommentare anpassen / korrigieren (Sarah)
- Mehr USA Tweets herunterladen (Sarah)
- Karte mit mehr USA Tweets erstellen (Marinko)
- README überarbeiten (Daniela)
- Main Methode Verlauf (Marinko)

                  

Following you find the current state of the project ordered by the timeline in which the individual steps where implemented.

#### Timeline 

![Timeline](/src/data/timeline.png)


#### Twitter API
- find Python library to work with Twitter API &rightarrow; [Tweepy](https://www.tweepy.org/ "Tweepy home")
- create credentials on Twitter developer portal to use the API
- become familiar to retrieve tweets by using library
- program python modul to receive data from the API

#### Collect labeled data sets (12/27/2020) (mainly responsible: Marinko Balikic, Sarah Bopp)
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
    
(mainly responsible Fabio Gebhard)    
    
Originally converting the data with TF-IDF to make it ready to use for ML-algorithms was also part of the standard preprocessing that we wnated to use, but in order to be able to compare different weighting methods we decided to make this part of the classifer folder and use two methods for weighting: TFIDF and ConutVectorizer 


#### Train Classifier (01/18/2021) 
- train different classifier based on the train set (75% of the data set)
  - classifier:  
       - Support-Vector-Machine (SVM) (mainly responsible: Marinko Balikic) 
           - using sklearn   
           - own implemenation without using sklearn for classifier  
       - Decision Tree Classifier / Random- Forest Classifier (mainly responsible: Sarah Bopp)
       - Logistic Regression (mainly responsible: Daniela Tratz-Weinmann)    
           - using sklearn   
           - own implemenation without using sklearn for classifier  
       - Ensemble Classifier (mainly responsible: Fabio Gebhard)
                 
#### Test Classifier (01/25/2021) 
- evaluate the different classifiers with the test set
- run m-fold cross-validation to determine the classifier with the smallest error
  - in addition to cross validation we want to use the F1-Score
  
#### Selection of meaninful tweets (02/01/2021) (mainly responsible: Sarah Bopp)
- select tweets from the twitter API which are:
  - english speech
  - located in United States (USA)
  - which were released in a specific time periode (e.g. time of US election)
  - tweets from a representable amount of people: Details in Data Section    
   
    

#### Analyze Data (Tweets) (02/08/2021) 
- execute preprocessing on selected twitter data
- use evaluated classifier to predict label of incoming tweets

#### Representation of data (02/15/2021) 
- plot data in an appropiate way

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
+ function for evaluating the classifiers

The second package is the package src.data, which is meant to contain both IDs of the Tweets that build the basis of our training data and the processed data in form of a csv file. Furthermore, it should contain the data sets with directly the texts of the Tweets instead of the IDs as csv files. 

The third package is the package src.classifiers (will be added later, when implementing the classifiers), which is meant to be responsible for the following functionalities:
+ Support-Vector-Machine (SVM)  
+ Decision Tree Classifier    
+ Random- Forest Classifier   
+ Logistic Regression   
+ Ensemble Classifer


#### Tests 
Tests for all core functionalities are be provided in the separate package tests.

##### Testing strategy

We decided to test all core functionalities of our project, which means we tested almost all or classes with only a few exceptions.   
First of all, we test all classes from the classifier package and check whether impelmmented exceotions are triggered and if all return types are correct. The acutual performance of each classfier is tested with several experiments (see Experiments section)
The only class from the classifier package that is not tested is svmclassifier_scratch, because we excluded it from our experiments (see Experiments section for Details) and thus the class is not part of the main functionalities of our package and testing it would only take up a lot of time without actually helping to possibly improve our project. logistic_regression_scratch, however is tested, eventhough it could be excluded from testing with the same reasoning, but since the duartion for testing this class can be easily reduced by using only a low number of iterations, we decided that we would include.

Regarding the utils package we also test the classes necessary for the main functionality. First of all we test the classes responsible for creating our datasets and for balancing our dataset. Again we mainly check the structure, return types and, in case of the dataframes, sizes of the return values. We also test the class reponsible for the calculation of the hate-speech-ratio, that is used for the examplary application of our classifier.
Some classes are excluded from testing: We did not test the corssvalidation function, since it is mainly a wrapper for the execution of our classifers,that are already tested seperatly,  for printing their results. With similar reasoning we did not test the test_map class, since it also only provides a visualization of our already tested classifier results. The other classes that we did not test were the classes for downloading data from the twitter api, because here we only use functions provide by twitter and the results can be directly checked by looking at the returned data, similary we did not test the class that provides functions for saving and loading the model because here we rely on pre-provided funtions from the pickle-library.


 ##### Unit Test Code Coverage
 
 The code coverage of the implemented classifiers:
 
 ![Tweets availability](/src/data/test_coverage_classifier.PNG)
 
 The code coverage of the utils scripts:
 
 ![Tweets availability](/src/data/test_coverage_utils.PNG)



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
 
 
 ## Experiments   
 
 ### Decision between from scratch methods (for Logistic Regression and SVM) and methods from sklearn   
We compared both from-scratch-methods (logistic_regression_scratch and svmclassifier_scratch) with the respective sklearn implementations, using a simple confusion matrix and the accuracy, recall and precision scores that we obtained by training and testing the classifiers on our data. As expected the performance of the from-scratch-methods was worse than the performance of sklearn methods, the biggest issue being the execution time (depending on iterations and data size 60-120 minutes longer) and in case of the SVM-Classifier the necessary storage capacity. The exact differences vary depending which parameters and which data set were used, but we did not find any setup in which the from-scratch-methods would perform better or even close to the sklearn implementations (these results can also be reproduced using the provided classifier methods in the clasifeir packkage and for evaluation the test_map function from the utils package).

Given these reuslts we decided to exlcude the from-scratch-methods from the following experiments.

 
 ### Final preprocessed data
 
 Combined dataset for binary classifier: Normal vs Hate Speech
 
 Imbalanced dataset:
 
 ![Tweets availability](/src/data/tweets_per_label_imbalanced_final.png)
 
 Balanced dataset:
 
 ![Tweets availability](/src/data/tweets_per_label_balanced_final.png)
 
 ### Comparison between imbalanced and balanced training dataset
 #### Results (10-fold cross validation) imbalanced dataset
 
  - Avg error SVM: 0.07519686970175847 <br>
  - Avg error Decision Tree: 0.09850973689058529 <br>
  - Avg error Random Forrest: 0.07165568518406577 <br>
  - Avg error Logistic Regression: 0.1004010615912931 <br>
  - Avg error Ensemble: 0.07165568518406577 <br>
 ----------------------
  - Avg accuracy SVM: 0.9248031302982416 <br>
  - Avg accuracy Decision Tree: 0.9014902631094147 <br>
  - Avg accuracy Random Forrest: 0.9283443148159343 <br>
  - Avg accuracy Logistic Regression: 0.8995989384087067 <br>
  - Avg accuracy Ensemble: 0.9283443148159343 <br>
 ----------------------
  - Avg precision SVM: 0.7466580822869207 <br>
  - Avg precision Decision Tree: 0.6047230314867805 <br>
  - Avg precision Random Forrest: 0.8083788280464453 <br>
  - Avg precision Logistic Regression: 0.5719471859967772 <br>
  - Avg precision Ensemble: 0.8083788280464453 <br>
 ----------------------
  - Avg recall SVM: 0.588171909682303 <br>
  - Avg recall Decision Tree: 0.5748100965857547 <br>
  - Avg recall Random Forrest: 0.5468931216260461 <br>
  - Avg recall Logistic Regression: 0.7316840880521852 <br>
  - Avg recall Ensemble: 0.5468931216260461 <br>
 

 #### Results (10-fold cross validation) balanced dataset
 
 - Avg error SVM: 0.1940567066521265 <br>
 - Avg error Decision Tree: 0.21859323882224646 <br>
 - Avg error Random Forrest: 0.18200654307524536 <br>
 - Avg error Logistic Regression: 0.171319520174482 <br>
 - Avg error Ensemble: 0.18200654307524536 <br> 
 ----------------------
 - Avg accuracy SVM: 0.8059432933478735 <br>
 - Avg accuracy Decision Tree: 0.7814067611777535 <br>
 - Avg accuracy Random Forrest: 0.8179934569247547 <br>
 - Avg accuracy Logistic Regression: 0.828680479825518 <br>
 - Avg accuracy Ensemble: 0.8179934569247547 <br>
 ----------------------
 - Avg precision SVM: 0.8116767725214927 <br>
 - Avg precision Decision Tree: 0.7811049109409147 <br>
 - Avg precision Random Forrest: 0.8212260290537614 <br>
 - Avg precision Logistic Regression: 0.8582117881409733 <br>
 - Avg precision Ensemble: 0.8212260290537614 <br>
 ----------------------
 - Avg recall SVM: 0.796718863865422 <br>
 - Avg recall Decision Tree: 0.7818837274030175 <br> 
 - Avg recall Random Forrest: 0.8130145051097839 <br>
 - Avg recall Logistic Regression: 0.7874117388925284 <br>
 - Avg recall Ensemble: 0.8130145051097839 <br>
 
 ### Comprison between preprocessing-methods:
 
 #### normal preprocessing method:

- Avg error SVM: 0.19626335156816574 <br>
- Avg error Decision Tree: 0.22287256806725697 <br>
- Avg error Random Forrest: 0.18555347640663378 <br>
- Avg error Logistic Regression: 0.1757183061601502 <br>
- Avg error Ensemble: 0.17604641376655236 <br>
----------------------
- Avg accuracy SVM: 0.8037366484318342 <br>
- Avg accuracy Decision Tree: 0.777127431932743 <br>
- Avg accuracy Random Forrest: 0.8144465235933662 <br>
- Avg accuracy Logistic Regression: 0.8242816938398498 <br>
- Avg accuracy Ensemble: 0.8239535862334477 <br>
----------------------
- Avg precision SVM: 0.8093553103221389 <br>
- Avg precision Decision Tree: 0.7783486639626233 <br>
- Avg precision Random Forrest: 0.8194140680878471 <br>
- Avg precision Logistic Regression: 0.8534382082078651 <br>
- Avg precision Ensemble: 0.8459772787293709 <br>
----------------------
- Avg recall SVM: 0.7948433600137679 <br>
- Avg recall Decision Tree: 0.7752060240897344 <br>
- Avg recall Random Forrest: 0.8065528025545291 <br>
- Avg recall Logistic Regression: 0.7828394538876038 <br>
- Avg recall Ensemble: 0.7920669863105279 <br>


#### restricted preprocessing:

- Avg error SVM: 0.16517281308849113 <br>
- Avg error Decision Tree: 0.17276790430741956 <br>
- Avg error Random Forrest: 0.15626720147549936 <br>
- Avg error Logistic Regression: 0.15047541878933846 <br>
- Avg error Ensemble: 0.14845359071008407 <br>
----------------------
- Avg accuracy SVM: 0.8348271869115088 <br>
- Avg accuracy Decision Tree: 0.8272320956925805 <br>
- Avg accuracy Random Forrest: 0.8437327985245006 <br>
- Avg accuracy Logistic Regression: 0.8495245812106615 <br>
- Avg accuracy Ensemble: 0.851546409289916 <br>
----------------------
- Avg precision SVM: 0.8453375116865516 <br>
- Avg precision Decision Tree: 0.8482504349312057 <br>
- Avg precision Random Forrest: 0.8676780161941668 <br>
- Avg precision Logistic Regression: 0.8812210994092959 <br>
- Avg precision Ensemble: 0.8811853070043323 <br>
----------------------
- Avg recall SVM: 0.8191968729989216 <br>
- Avg recall Decision Tree: 0.7968548740212839 <br>
- Avg recall Random Forrest: 0.8110103111670066 <br>
- Avg recall Logistic Regression: 0.8076247964111127 <br>
- Avg recall Ensemble: 0.812296533100582 <br>

-> Choose restricted preprocessing method


 ### Comparison using restricted preprocessing & balanced dataset for all 4 combinations: CountVectorizer | TfIdf - unigrams | unigrams & bigrams:
 
 #### Option 1:  -> CountVectorizer + unigrams

- Avg error SVM: 0.16657579062159217 <br>
- Avg error Decision Tree: 0.17780806979280264 <br>
- Avg error Random Forrest: 0.15659760087241004 <br>
- Avg error Logistic Regression: 0.15152671755725192 <br>
- Avg error Ensemble: 0.14918211559432934 <br>
----------------------
- Avg accuracy SVM: 0.8334242093784079 <br>
- Avg accuracy Decision Tree: 0.8221919302071974 <br>
- Avg accuracy Random Forrest: 0.8434023991275899 <br>
- Avg accuracy Logistic Regression: 0.8484732824427482 <br>
- Avg accuracy Ensemble: 0.8508178844056706 <br>
----------------------
- Avg precision SVM: 0.8431290270480719 <br>
- Avg precision Decision Tree: 0.8405728429437683 <br>
- Avg precision Random Forrest: 0.8715008501422272 <br>
- Avg precision Logistic Regression: 0.8811684281627038 <br>
- Avg precision Ensemble: 0.8810063708250233 <br>
----------------------
- Avg recall SVM: 0.8190693533478186 <br>
- Avg recall Decision Tree: 0.7951153334879827 <br>
- Avg recall Random Forrest: 0.8055459766199009 <br>
- Avg recall Logistic Regression: 0.805507181794173 <br>
- Avg recall Ensemble: 0.8110359864449526 <br>
----------------------
----------------------
- F1 Score SVM: 0.8309250629982712 <br>
- F1 Score Decision Tree: 0.8172124320127871 <br>
- F1 Score Random Forrest: 0.8372264772216413 <br>
- F1 Score Logistic Regression: 0.8416407909917994 <br>
- F1 Score Ensemble: 0.8445744492817719 <br>

![Table Option1](/src/data/option1_table.png)
![Option1 Bar](/src/data/option1.png)

#### Option 2:  -> CountVectorizer + unigrams&bigrams

- Avg error SVM: 0.14923664122137403 <br>
- Avg error Decision Tree: 0.17093784078516902 <br>
- Avg error Random Forrest: 0.15932388222464555 <br>
- Avg error Logistic Regression: 0.1504907306434024 <br>
- Avg error Ensemble: 0.1489094874591058 <br>
----------------------
- Avg accuracy SVM: 0.8507633587786259 <br>
- Avg accuracy Decision Tree: 0.8290621592148311 <br>
- Avg accuracy Random Forrest: 0.8406761177753544 <br>
- Avg accuracy Logistic Regression: 0.8495092693565975 <br>
- Avg accuracy Ensemble: 0.8510905125408943 <br>
----------------------
- Avg precision SVM: 0.8860824907667777 <br>
- Avg precision Decision Tree: 0.8487757851759218 <br>
- Avg precision Random Forrest: 0.8804947810019144 <br>
- Avg precision Logistic Regression: 0.8939233518779457 <br>
- Avg precision Ensemble: 0.8935484602185614 <br>
----------------------
- Avg recall SVM: 0.8049453092395706 <br>
- Avg recall Decision Tree: 0.8008090880023777 <br>
- Avg recall Random Forrest: 0.7883257745438029 <br>
- Avg recall Logistic Regression: 0.7929642982009459 <br>
- Avg recall Ensemble: 0.7969882544445406 <br>
----------------------
----------------------
- F1 Score SVM: 0.8435673790098013 <br>
- F1 Score Decision Tree: 0.8240950477869279 <br>
- F1 Score Random Forrest: 0.8318650293567708 <br>
- F1 Score Logistic Regression: 0.8404226604352475 <br>
- F1 Score Ensemble: 0.8425106907105752 <br>

![Table Option2](/src/data/option2_table.png)
![Option2 Bar](/src/data/option2.png)

#### Option 3:  -> TFIDF + unigrams

- Avg error SVM: 0.14585605234460197 <br>
- Avg error Decision Tree: 0.20408942202835334 <br>
- Avg error Random Forrest: 0.17104689203925846 <br>
- Avg error Logistic Regression: 0.15441657579062157 <br>
- Avg error Ensemble: 0.14874591057797165 <br>
----------------------
- Avg accuracy SVM: 0.854143947655398 <br>
- Avg accuracy Decision Tree: 0.7959105779716468 <br>
- Avg accuracy Random Forrest: 0.8289531079607416 <br>
- Avg accuracy Logistic Regression: 0.8455834242093783 <br>
- Avg accuracy Ensemble: 0.8512540894220283 <br>
----------------------
- Avg precision SVM: 0.8795617927693531 <br>
- Avg precision Decision Tree: 0.7870830760505199 <br>
- Avg precision Random Forrest: 0.8204297260343305 <br>
- Avg precision Logistic Regression: 0.8867509234461292 <br>
- Avg precision Ensemble: 0.879470836496006 <br>
----------------------
- Avg recall SVM: 0.8204089657136766 <br>
- Avg recall Decision Tree: 0.8114804204158437 <br>
- Avg recall Random Forrest: 0.842133438434096 <br>
- Avg recall Logistic Regression: 0.7921347242721999 <br>
- Avg recall Ensemble: 0.8138447214014068 <br>
----------------------
----------------------
- F1 Score SVM: 0.8489562271425102 <br>
- F1 Score Decision Tree: 0.7990955715772666 <br>
- F1 Score Random Forrest: 0.8311399181031899 <br>
- F1 Score Logistic Regression: 0.8367767026857877 <br>
- F1 Score Ensemble: 0.8453860765296495 <br>

![Table Option3](/src/data/option3_table.png)
![Option3 Bar](/src/data/option3.png)

#### Option 4:  -> TFIDF + unigrams&bigrams

- Avg error SVM: 0.1489640130861505 <br>
- Avg error Decision Tree: 0.22791712104689205 <br>
- Avg error Random Forrest: 0.18675027262813523 <br>
- Avg error Logistic Regression: 0.16079607415485278 <br>
- Avg error Ensemble: 0.1549618320610687 <br>
----------------------
- Avg accuracy SVM: 0.8510359869138495 <br>
- Avg accuracy Decision Tree: 0.772082878953108 <br>
- Avg accuracy Random Forrest: 0.8132497273718646 <br>
- Avg accuracy Logistic Regression: 0.8392039258451472 <br>
- Avg accuracy Ensemble: 0.8450381679389313 <br>
----------------------
- Avg precision SVM: 0.8559000422394357 <br>
- Avg precision Decision Tree: 0.7547564870886946 <br>
- Avg precision Random Forrest: 0.7888076360757029 <br>
- Avg precision Logistic Regression: 0.8585147461136706 <br>
- Avg precision Ensemble: 0.8520273300275368 <br>
----------------------
- Avg recall SVM: 0.8439628790303093 <br>
- Avg recall Decision Tree: 0.8062086510203162 <br>
- Avg recall Random Forrest: 0.8556716536878483 <br>
- Avg recall Logistic Regression: 0.8122046854834031 <br>
- Avg recall Ensemble: 0.8349085059652281 <br>
----------------------
----------------------
- F1 Score SVM: 0.849889546706491 <br>
- F1 Score Decision Tree: 0.7796345920213817 <br>
- F1 Score Random Forrest: 0.8208803097782366 <br>
- F1 Score Logistic Regression: 0.8347178899853515 <br>
- F1 Score Ensemble: 0.8433810581019444 <br>

![Table Option4](/src/data/option4_table.png)
![Option4 Bar](/src/data/option4.png)


Comparing the F1 scores for all options, you can identify the SVM Classifier of Option 4 as the best one, so using TF-IDF plus unigrams and bigrams:    
![Table F1 scores](/src/data/f1_score_table.png)
![F1 score Bar](/src/data/f1_score.png)



    
      

 ### Retrieved data set for exemplary application of classifiers
 ![Distribution of tweets over cities](/src/data/statistics_all.png)
 In order to show an exemplary application of our hate speech classifiers, we queried tweets posted on the 12th and 15th February, 2021 in the 50 largest cities per US state. Above you can see the distribution of tweets per city.
 
 
 
 ### USA Hate Speech Map (~1000 tweets per US state) 
 
 ![Tweets availability](/src/data/USA_Hate_Speech_Map_1000_tweets_per_state.png)
 
  ### USA Hate Speech Map (3000-7000 tweets per US state) 
 
 ![Tweets availability](/src/data/USA_Hate_Speech_Map_All_Tweet.png)
 
 
 
 

 

