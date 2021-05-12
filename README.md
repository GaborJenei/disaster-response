# disaster-response
ETL &amp; ML Pipeline to classify messages related to disasters.

The intention and motivation is to help essential disaster response services to filter through a myriad of messages identifying the relevant one.

This Disaster Response Messages project builds on text messages from Figure8.
The text data is extracted and cleaned through the ETL pipeline before it's sent through the NLP pipeline.
The processed text data and corresponding disaster categories are used to train a classification model

The final output is a web app, that lets the user to input a message and instantly get a possible category based on training data.

The F1 score across all categories is 0.69.

Sections:
- [App Gallery](#App-Gallery)
- [Pipeline Process](#Pipeline-Process)
    - [Data](#Data)
    - [Cleaning and Processing](#Cleaning-and-Processing)  
    - [Classifier](#Classifier)
    
- [User Guide](#User-Guide)
    - [Installation](#Installation)
    - [Usage](#Usage)

- [File Descriptions](#File-Descriptions)  
- [Licensing, Authors, Acknowledgements](#Licensing,-Authors,-Acknowledgements)  
- [Packages used](#Packages-used)


## App Gallery


# Pipeline Process
## Data
Three randomly selected message and their categories:
_Message 1:_
> 'hurry hurry send water and food for people, please I cant take it anymore, sos'  
> related, request, aid_related, water, food, direct_report  

_Message 2:_  
> 'Want to know what the government said about the earthquake '  
> related, weather_related, earthquake

_Message 3:_  
> 'Local officials appealed for equipment and other items to assist resue efforts.'  
> related
 

The Categories:
> 'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security',
> 'military', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
> 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers',
> 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'

## Cleaning and Processing
Text data is challenging, the cleaning process and tokenization were refined to achieve a better model performance
__Cleaning__
- Remove duplicates
- Remove non-English languages

### Feature Extraction

__Tokenizations__  
1. Text is normalised to lower case
2. Cleaning with regex
    - urls (including some broken ones with white space in them) were replaced with 'url_placeholder' string
    - email addresses replaced with 'email' string
    - all @mention was replaced with 'at_mention' string
    - retweet notations were replaced with 're_tweet' string
    - All non-alphanumeric characters replaced with ' ' (white space)
3. Message text converted to word tokens (nltk `word_tokenizer`)
4. Stopwords removed (`nltk.corpus.stopwords.words('english')`)
5. Tokens Lemmatized (nltk.stem.wordnet `WordNetLemmatizer`)
6. Tokens Stemmed

__TF-IDF__
Term Frequency — Inverse Document Frequency transformation was applied on the tokenized training data.

__Word Count__  
The number of words (scaled to 0-1) of each message was added as an additional feature to improve the model accuracy. 


Training Notes

# TODO add a figure of the vocabulary size 

## Classifier
Support vector machine classifier was used for this project. 
The classifier was implemented in scikit-learns Pipeline object with the feature extraction process as a FuatureUnion.
GridSearchCV was used to run the training with using a range of parameters

The parameter space the GridSearchCV searched:
- Kernel type of Support Vector Classifier: 'linear', 'poly', 'rbf', 'sigmoid'
- Regularization parameter C: 0.5, 1.0, 2.5, 5

The degree of the polynomial kernel function was also considered, however it was tested separately, because this parameter is only considered by when `kernel='poly'` and ignored for all other kernels by SVC.
At the same time GridSearchCV methodically trains avery possible combinations of degree and kernel scaling up the runtimes significantly.
 
# User Guide
## Installation
- Download or clone this repository to your local machine
- (optional) create a new virtual environment if you wish to do so and activate it (here are the details with [conda](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html) and [pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/))
- Install the dependencies from this repository requirements.txt
    - Conda: `conda install --file requirements.txt`
    - Pip: `pip install -r requirements.txt`

## Usage
1. cd into data folder, run ETL pipeline transforming, cleaning and putting the data into an SQL database:
    `python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db`  
2. cd into ML_Pipeline folder, then run ML pipeline training the classification model and saving it into a pickle file:  
    `python train_classifier.py disaster_response.db saved_model.pkl`  
    :warning: Using the full GridSearchCV parameters can take long time even on modern machines  
3. cd into app folder, then run the web app:  
    `python run.py`  
4. Go to http://0.0.0.0:3001/ Alternatively, type http://localhost:3001/ in browser  

## File Descriptions
__data/__  
    _disaster_categories.csv_:  input datafile of categories  
    _disaster_messages.csv_:    input datafile of messages  
    _disaster_response.db_:     database of cleaned message data  
    _process_data.py_:          ETL pipeline extracting from the two csv, transforming, cleaning and loading data into the db  
__ML_Pipeline/__  
    _NLP_pipeline.py_:          script file holding tokenizer and message length transformer class  
    _train_classifier.py_:      ML Pipeline definition and training  
    _saved_model.pkl_:          saved trained ML model  
__app/__  
    _run.py_:                   Flask app, backend of the webapp  
    _NLP_pipeline.py_:          script file holding tokenizer and message length transformer class  
    __templates/__
        _master.html_:          landing page of the web app  
        _go.html_:              page rendering the classifier output, with form for another message to be classified  

## Licensing, Authors, Acknowledgements 
- MIT License
- Thanks for Figure8 for the data
- Thanks for Udacity for the frame of the web app

## Packages used
Libraries required for the app to run are listed in the requirement.txt file
- Numpy, Pandas, SQLite, SQLAlchemy
- langdetect, re (regex), NLTK, Scikit Learn
- Plotly, Flask


