# CSE251b Final Project - Social Media Sentiment Analysis

## Authors

- **Gabrielle Rackner** (HDSI) - [grackner@ucsd.edu](mailto:grackner@ucsd.edu)  
- **Kira Fleischer** (HDSI) - [kfleisch@ucsd.edu](mailto:kfleisch@ucsd.edu)  
- **Kevin Wong** (HDSI) - [kew024@ucsd.edu](mailto:kew024@ucsd.edu)  
- **Krishna Priyanka Ponnaganti** (CSE) - [kponnaganti@ucsd.edu](mailto:kponnaganti@ucsd.edu)  
- **Atefeh Mollabagher** (ECE) - [amollabagher@ucsd.edu](mailto:amollabagher@ucsd.edu)  
- **Mihir Joshi** (HDSI) - [myjoshi@ucsd.edu](mailto:myjoshi@ucsd.edu)  
- **Rohil Kadekar** (CSE) - [rkadekar@ucsd.edu](mailto:rkadekar@ucsd.edu)  

## Overview

Our project aims to analyze social media sentiments using the pretrained models **BERT** and **RoBERTa**. Sentiment analysis in this context involves classifying the underlying emotion of a given social media post to understand the type of content frequently shared on these platforms.  

Our analysis will focus on evaluating which pretrained models perform best for sentiment classification tasks. Additionally, we will compare the performance of different fine-tuning methods on these models. We aim to apply these models to a labeled **Reddit post dataset** and assess the overall sentiment distribution on the platform.  

## Dataset

We will be using the **GoEmotions dataset** from Kaggle ([Chanda, 2021](https://www.kaggle.com/datasets/Chanda_2021)). This dataset consists of **58,009** Reddit comments, each labeled with one of **28 different emotions**.

## Network Architectures

We will be testing four models based on **BERT** and **RoBERTa**:

1. **BERT** - A contextual embedding model trained on diverse text sources.  
2. **RoBERTa** - A larger version of BERT, trained for longer on a more extensive dataset, outperforming BERT.  
3. **Social Media BERT** - A variant of BERT pre-trained specifically on social media data for sentiment analysis ([NLP Town, 2023](https://huggingface.co/nlp_town)).  
4. **Social Media RoBERTa** - A version of RoBERTa fine-tuned using the **TwitterEval benchmark** ([RoBERTa Sentiment, 2023](https://huggingface.co/roberta_sentiment)).  

If **RoBERTa** is too computationally demanding, we will use **DistilBERT** ([DistilBERT, 2019](https://arxiv.org/abs/1910.01108)), a smaller, more efficient version of BERT.  

We plan to download these models from **Hugging Face** and run them on a **GPU in the UCSD DataHub environment**.
