#!/bin/bash
# setup environment
if [[ "$OSTYPE" == "msys" ]]; then
    echo "windows OS detected"
    conda env create -f ./environment_win.yml
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
    echo "linux OS deteced"
    conda env create -f ./environment.yml
fi

conda init bash
conda activate sent_env

# --or--
# conda create --name sent_env --file requirements.txt


mkdir -p ./data

# then download the sql files:
gdown --fuzzy https://drive.google.com/file/d/1YG_AQIbcY2Mi-bKMLN1hff66jDBFeh4O/view?usp=sharing -O ./data/spx_news_sentiment_fundamental.db
gdown --fuzzy https://drive.google.com/file/d/1C49ElctSD0hPsukQTkioneA7PeWBlMfe/view?usp=sharing -O ./data/spx_news_sentiment_price.db
