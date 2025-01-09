# pydigest-classifier
pythondigest.ru is a popular Russian Python newsletter.
This project aims to automate time-consuming process of selecting articles that are good enough for publication by means of machine learning.

# Structure
Classifier is trained and dumped remotely, then it can be accessed via POST requests to the provided API (see API section).

# API
API is a simplistic Flask app, currently it handles POST requests to ```/api/v1.0/classify```.
Each POST request should contain a json object of following structure:
```
{
    "links": [{
        "link": "http://www.python.org/dev/peps/pep-0453/",
        "data": {
            "title": "Pip будет включен в поставку Python 3.4",
            "description": "Это решение из PEP одобрено 22 октября 2013 г",
            "language": "en",
            "article": "FULL HTML SOURCE CODE OF THIS ARTICLE"
        }
    }, {
        "link": "http://www.python.org/dev/peps/pep-0453/",
        "data": {
            "title": "Pip будет включен в поставку Python 3.4",
            "description": "Это решение из PEP одобрено 22 октября 2013 г",
            "language": "en",
            "article": "FULL HTML SOURCE CODE OF THIS ARTICLE"
        }
    }]
}
```
Response JSON will be
```
{
    "links": [{"http://www.python.org/dev/peps/pep-0453/": True},
              {"http://www.python.org/dev/peps/pep-0453/": True}]
}
```
Where dict value is a prediction for according link.

Classifier is uploaded from ```classifier_64.pkl``` file, it must be placed in the working directory of API script.

# Classifier
There are four classifier-related scripts: ```classifier.py```, ```train.py```, ```vectorizer.py```, ```text_metrics.py```.
 -  ```classifier.py``` - This file contains classifier itself (compliant with scikit-learn style, ready to use with Pipelines) and some minor utility functions
 -  ```vectorizer.py``` - Custom vectorizer for JSON input is here. See API sections for details on input format.
 -  ```train.py``` - Trains the classifier and dumps it for future use. Accepts two command line args, first one is path to the folder with training files, second one - dump location
 -  ```text_metrics.py``` - This file contains some utility functions to extract geometric properties from texts

# Quality report
Use ```report.py``` for quality checks. Accepts one command line arg - path to input file. Report file is a csv file with named columns. Names are: ```["link", "classificator", "moderator"]```, order is arbitrary. Reporting script computes various metrics (accuracy, precision, recall, F1) and plots confusion matrix, based on input file.
