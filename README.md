# Text Analytics semestral work
We are working with a dataset of various [reviews](https://www.kaggle.com/datasets/trainingdatapro/6000-messengers-reviews-google-play) of different applications on Google Play.

The main file is *text_analytics_processing_and_model.ipynb*, which loads the data, builds a pipeline, exports usable data and trains a language model.

The primary objective of this project was to build and train a language model that can accurately detect sentiment in reviews and predict whether a review is positive or negative.

The language model built in this project is designed to detect sentiment in reviews, categorizing them as either positive or negative. The model tries to provide insights into the overall sentiment of user reviews for applications on Google Play.

![image](https://github.com/Yes-and/text-analytics-semestral-work/assets/72066894/a9d9d724-baf4-48ac-9e37-f5ca37dacdc7)

Another dataset of Threads reviews on Google Play was added, to enhance the accuracy of the model. The dataset is available [here](https://www.kaggle.com/datasets/jayagopal20/threads-app-reviews-dataset)

# Usage

## Competitive Market Analysis

Analysis of user reviews and sentiments to understand the strengths and weaknesses of the competitive product.

## Social Media Perception Measurement

The project allows us to measure and understand the overall perception of a company or brand on social media.

# Data structure
![image](https://github.com/Yes-and/text-analytics-semestral-work/assets/72066894/666c93d2-a357-4b5e-9095-c8e16919dda5)

![image](https://github.com/Yes-and/text-analytics-semestral-work/assets/72066894/f59546c5-7e59-4950-9c18-148b55dc793a)

![image](https://github.com/Yes-and/text-analytics-semestral-work/assets/72066894/70d61230-87d6-4637-8bca-98c17e89da6d)

![image](https://github.com/Yes-and/text-analytics-semestral-work/assets/72066894/35a2a720-11cf-423e-a216-649879da2952)

# Dependencies

## spaCy

A natural language processing library used for various linguistic tasks such as part-of-speech tagging, dependency parsing, and named entity recognition.

## langdetect

A language detection library used for identifying the language of the text.

## Flask

A web framework used for building the RESTful API that exposes the text analytics functionalities.

## requests

A library for making HTTP requests, used in the part of the code where a request is made to the prediction service.

## pandas

A data manipulation library, which might be used for handling and processing datasets, although it's not explicitly shown in the provided code
