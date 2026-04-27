# Renegade Rehab AI Lab Demo Video Script

## Opening

Hi, my name is Caylem, and this is Renegade Rehab AI Lab, a local AI sport performance prototype I built on my Mac mini.

I created this project to demonstrate my interest in artificial intelligence, sport science, movement science, and human performance.

## Project Goal

The goal of Renegade Rehab AI Lab is to show how wearable movement data and heart-rate data can be transformed into useful sport science insights.

This project uses machine learning to classify physical activity and then uses a local AI model to generate a plain-English summary that could be useful for coaches, researchers, or performance professionals.

## Dataset

For this project, I used the PAMAP2 Physical Activity Monitoring dataset.

This dataset includes wearable sensor and heart-rate data from subjects performing activities such as walking, running, cycling, sitting, standing, stair climbing, and other movement tasks.

## Workflow

The project starts with raw sensor data.

I created a data pipeline that cleans the data, handles missing values, and turns raw movement readings into activity windows.

Then I trained a machine-learning model to classify the activity being performed based on accelerometer, gyroscope, and heart-rate features.

## Dashboard

This dashboard shows the results.

Here, I can select a subject and view the number of activity windows, the number of activities detected, average heart rate, maximum heart rate, activity distribution, heart-rate trends, and a sample of the model’s predictions.

## Local AI Summary

One important part of this project is that the AI summary runs locally on my Mac mini using Ollama.

The dashboard sends a structured summary of the subject’s activity and heart-rate data to the local model. The model then generates a coach-facing sport science interpretation.

This keeps the workflow local and demonstrates how AI tools can support data interpretation without relying entirely on cloud-based subscription tools.

## Limitations

This is a proof of concept. It is not a medical device. It does not diagnose injury, prescribe training, or replace professional judgment.

The goal is to demonstrate an applied AI workflow for sport and movement science.

## Closing

Building Renegade Rehab AI Lab helped me practice data cleaning, feature engineering, machine learning, model evaluation, dashboard development, local AI integration, and applied communication.

This project reflects the kind of work I hope to continue in graduate study: using AI and data analysis to support human performance, movement assessment, and evidence-based sport science.
