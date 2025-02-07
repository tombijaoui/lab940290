# Data Management & Collection Lab - On Target 

## Not to lie, but to emphasize

### Jornet Jeremy - Sasson Eden - Bijaoui Tom

Technion - Israel Institute of Technology - Faculty of Data Science and Decisions

![](https://upload.wikimedia.org/wikipedia/commons/b/b7/Technion_logo.svg)


## Project's Files

* Final Project Report
* Model Interpretability
    * Model Interpretability Notebook
    * Model Interpretability Results
* On Target tool
    * On Target Notebook
    * Example generated instructions
* Scraping
    * Scraping "Comparably"
    * Scraping companies' websites
* Requirements


## Overview
First impressions and emphasizing the right qualities are crucial in the business world, especially when applying for a job. That's why we developed On Target, our revolutionary big data tool. This system generates tailored guidelines that enable candidates to highlight the skills and values most valued by the specific company they are targeting using Statistical and Machine Learning methods.

## Implementing Methods
* Data Preprocessing
* Features Engineering
* Pre-trained models from Hugging Face]
* Statistical tests to significance inference
* Machine Learning model to Model Interpretability
* NLP keywords extraction techniques
* LLM to generate instructions

## Model Interpretability
The model interpretability notebook gives information about the importance of the features in the recruitment process by training for each company a Random Forest model for a binary classification task.

### Running the code
Before you begin, make sure the following prerequisites are met:

* _Databricks Account_: A Databricks account is required to run this project.
* _Databricks Cluster_: A cluster must be configured and started before running the code.

Then, start the cluster and run the code. The file "Model Interpretability Results" provides the decreasing ordered list of the importance of each feature.

## On Target Tool
You can find here the main notebook with all our work. 

Here, we were able to learn the key values for each company, the inherent values of each profile, and understand their level of importance for each feature. Finally, the system generates the instructions the candidates has to follow to enhance his profil towards a specific company that he targeted.

### Running the code
Same as "Running the Code" section in the "Model Interpretability" section.

Then, each user enters his employee ID and the company's name that he targets. The system outputs the instructions. The file "Example generated instructions" provides a great example of the format of the instructions. 

## Scraping 
This file is about the scraping methods we used to scrape the relevant data from the companies' websites and from Comparably. For the scraping mission, we used the BrightData application that allows us to do high-scaling scraping without being blocked. 

### Running the code
Before you begin, make sure you have a BrightData account. 
Install all the dependencies thanks to the requirements files according to the chosen scraping task.
Copy paste the following lines of codes by replacing "username" and "password" by your own username password.

``` 
AUTH = 'username:password'
SBR_WEBDRIVER = f'https://{AUTH}@brd.superproxy.io:9515'
```

## Links

Inception alert 🚨 : You may check our Linkedin post about [On Target](https://www.linkedin.com/posts/tom-bijaoui-2799402ab_machinelearning-bigdata-nlp-activity-7293316200053248000-um9R?utm_source=share&utm_medium=member_ios&rcm=ACoAAEq2IX0Bx9yjkh8KcKEaqRrj5e5HWYojE1c) based on Linkedin Big Data!

