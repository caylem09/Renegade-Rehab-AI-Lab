# AthleteIQ: Local AI for Sport Performance Monitoring

## Project Overview

AthleteIQ is a local AI sport performance proof of concept designed to demonstrate how wearable movement data, heart-rate data, machine learning, and local language models can support sport science and human performance analysis.

The project was built on a Mac mini using Python, scikit-learn, Streamlit, Ollama, and a local language model. The goal was to create a practical applied AI system that connects data science with sport and movement science.

## Problem

Sport science professionals often work with large amounts of wearable sensor data, heart-rate data, and movement information. Raw data alone is difficult to interpret quickly. Coaches, researchers, and performance professionals need tools that can transform movement data into clear summaries that support decision-making.

AthleteIQ explores how machine learning and local AI can help turn raw wearable data into useful performance information.

## Dataset

AthleteIQ uses the PAMAP2 Physical Activity Monitoring dataset, which includes wearable sensor and heart-rate data from subjects performing multiple physical activities.

The dataset includes activities such as walking, running, cycling, sitting, standing, stair climbing, lying, and other movement tasks.

## Methods

The project workflow included:

1. Loading raw wearable sensor data
2. Cleaning and organizing the dataset
3. Handling missing heart-rate and movement values
4. Creating time-window features from raw sensor readings
5. Training a machine-learning model to classify physical activity
6. Evaluating model predictions
7. Building a Streamlit dashboard
8. Generating a local AI summary using Ollama

The machine-learning model uses engineered features from accelerometer, gyroscope, and heart-rate data to classify the activity being performed.

## Dashboard

The AthleteIQ dashboard allows the user to select a subject and view:

- Number of activity windows analyzed
- Number of activities detected
- Average heart rate
- Maximum heart rate
- Activity distribution
- Heart-rate trend
- Model prediction sample
- Model report
- Local AI sport science summary

## Local AI Component

AthleteIQ uses Ollama to run a local language model on the Mac mini. The dashboard sends a structured summary of the selected subject’s activity and heart-rate data to the local model.

The local model then generates a plain-English sport science summary. This demonstrates how AI can help translate technical model outputs into coach-facing interpretation while keeping the workflow local and privacy-conscious.

## Results

The proof of concept successfully demonstrates an end-to-end applied AI workflow:

Raw wearable data is transformed into cleaned data, engineered features, a trained activity-recognition model, a dashboard, and a local AI-generated interpretation.

This shows that a relatively small local system can support meaningful sport science data analysis without relying entirely on subscription-based cloud AI tools.

## Limitations

AthleteIQ is a proof of concept and is not a medical device.

It does not diagnose injury, prescribe training, or replace professional judgment. The AI-generated summary is intended to support interpretation, not make clinical or medical decisions.

The current version uses a public dataset and a basic machine-learning model. Future versions could include more advanced modeling, personalized baselines, additional physiological inputs, and more detailed performance metrics.

## Relevance to Graduate Study

AthleteIQ connects directly to my interest in artificial intelligence, sport science, movement science, and human performance.

The project helped me practice data cleaning, feature engineering, machine learning, model evaluation, dashboard development, local AI integration, and applied communication.

It also reflects the kind of graduate work I hope to pursue: using AI and data analysis to support evidence-based performance assessment, movement analysis, and human performance decision-making.

## Future Work

Future improvements could include:

- Adding training load calculations
- Adding subject-specific baselines
- Comparing multiple machine-learning models
- Adding feature importance visualizations
- Adding confusion matrix charts
- Exporting coach-facing PDF reports
- Adding wellness, sleep, or recovery inputs
- Creating a polished demo version for admissions review

