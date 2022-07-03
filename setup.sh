#!/bin/bash
# setup environment
conda create -f ./environment.yml

# --or--
# !conda create --name sent_env --file requirements.txt
conda activate sent_env

!mkdir -p ./data

# then download the sql files:
!gdown https://drive.google.com/file/d/1YG_AQIbcY2Mi-bKMLN1hff66jDBFeh4O/view?usp=sharing -O ./data/spx_news_sentiment_fundamental.db
!gdown https://drive.google.com/file/d/1C49ElctSD0hPsukQTkioneA7PeWBlMfe/view?usp=sharing -O ./data/spx_news_sentiment_price.db
