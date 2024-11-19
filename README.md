# Sleep Disorder Prediction: Data Science Project

## Project Overview

The goal of this data science project is to analyze various lifestyle and medical factors to predict the occurrence and type of sleep disorders individuals may experience. Sleep disorders, such as **Insomnia** and **Sleep Apnea**, can have serious consequences for one's health and overall well-being. By identifying individuals who are at risk, we can provide timely interventions and treatments to improve their sleep quality and, consequently, their health.

In this project, we focus on a dataset that includes variables like **age**, **BMI**, **physical activity**, **sleep duration**, and **blood pressure**, among others. Our aim is to use these features to build a model that can predict the likelihood of different types of sleep disorders, thereby offering a data-driven approach to improving sleep health.

## Dataset Description

The dataset used for this analysis is the **Sleep Health and Lifestyle Dataset**, which contains information about 400 individuals, with 13 columns representing various attributes related to sleep and daily habits. Below is an overview of the dataset's key features:

### Key Features:

#### 1. Comprehensive Sleep Metrics
This includes details like sleep duration, sleep quality, and other factors that may influence an individual's sleep patterns.

#### 2. Lifestyle Factors
This section provides insights into factors such as physical activity, stress levels, and BMI, which are likely to have an impact on sleep health.

#### 3. Cardiovascular Health
Key cardiovascular measurements such as blood pressure and heart rate are included, as these health indicators are often linked to sleep disorders.

#### 4. Sleep Disorder Analysis
The main objective of the project is to identify the presence or absence of sleep disorders. The **Sleep Disorder** column classifies individuals into one of the following categories:
   - **None**: No sleep disorder is detected.
   - **Insomnia**: Difficulty falling or staying asleep, leading to poor sleep quality.
   - **Sleep Apnea**: Interrupted breathing during sleep, which results in fragmented sleep patterns.

### Data Dictionary

| **Column Name**     | **Description**                                              |
|---------------------|--------------------------------------------------------------|
| `Person_ID`         | Unique identifier for each person                             |
| `Gender`            | Gender of the individual (Male/Female)                        |
| `Age`               | Age of the individual (in years)                              |
| `Occupation`        | Occupation of the individual                                  |
| `Sleep_duration`    | Hours of sleep per night                                      |
| `Quality_of_sleep`  | Subjective rating of sleep quality (scale from 1 to 10)       |
| `Physical_activity` | Level of physical activity (Low/Medium/High)                  |
| `Stress_Level`      | Subjective rating of stress level (scale from 1 to 10)       |
| `BMI_category`      | Body Mass Index (BMI) category (Underweight/Normal/Overweight/Obesity) |
| `Blood_pressure`    | Blood pressure (in mmHg)                                      |
| `Heart_rate`        | Heart rate (beats per minute)                                  |
| `Daily_Steps`       | Number of steps taken per day                                 |
| `Sleep_disorder`    | Type of sleep disorder (None, Insomnia, Sleep Apnea)          |

## Impact and Objective

This project aims to provide valuable insights into the factors that influence sleep disorders. By building a predictive model, we seek to identify individuals who may be at risk for these conditions. Understanding these relationships will enable health professionals to take proactive steps to improve sleep quality, ultimately contributing to better overall health and well-being for individuals affected by sleep disorders.
