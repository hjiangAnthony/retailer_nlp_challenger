# retailer_nlp_challenger

## Repo Summary
This project has a dataset of offers and some associated metadata around the retailers and brands that are sponsoring the offer. This project also comes with a dataset of some brands that we support on our platform and the categories that those products belong to.

In a nutshell, the approach to this goal is to:
1. being able to identify new entities (brand and retailer) and categories; more specifically, knowing how to train a new model that is capable of doing that
2. being able to filter existing offers that are available on the platform, and calculating similarity scores
3. being able to design a simple application and deploy it to the cloud; doing so requires the knowledge of developing not only Python scripts but also some familiarity with certain deployment environments (in this case it is HuggingFace)

Details of this project will be updated soon for confidentiality reasons.

## Demo
I have created a [Huggingface Space](https://huggingface.co/spaces/hjianganthony/fetch_ner) as my demo. 
<div align=center>
<img width="800" height="400" src="https://github.com/hjiangAnthony/retailer_nlp_challenger/blob/main/images/demo.gradio.png"/>
</div>

## Data
Three CSV files are provided initially: `categories.csv`, `brand_category.csv`, and `offer_retailer.csv`. 

## Methodology

### Goals
```
- If a user searches for a category (ex. diapers) the tool should return a list of offers that are relevant to that category
- If a user searches for a brand (ex. Huggies) the tool should return a list of offers that are relevant to that brand
- If a user searches for a retailer (ex. Target) the tool should return a list of offers that are relevant to that retailer
- The tool should also return the score that was used to measure the similarity of the text input with each offer
```

### Model Training and Results
To train the spaCy model to learn new entities, you can refer to the [spaCy template](https://github.com/explosion/projects) developed by Explosion. You will also be able to write a config file. One thing worth mentioning is that you will need a GPU to run my config file. Otherwise, please choose your configuration carefully.

#### Data Preparing
spaCy takes the [spaCy binary format](https://spacy.io/api/data-formats). One needs to transform the abovementioned files to labeled files (in my case it is `JSON`) and then convert them to spaCy corpus. Feel free to try other methods.

<div align=center>
<img width="800" height="400" src="https://github.com/hjiangAnthony/retailer_nlp_challenger/blob/main/images/spacy_ner_demo.png"/>
</div>

```
============================= Training pipeline =============================
ℹ Pipeline: ['transformer', 'ner']
ℹ Initial learn rate: 0.0
E    #       LOSS TRANS...  LOSS NER  ENTS_F  ENTS_P  ENTS_R  SCORE 
---  ------  -------------  --------  ------  ------  ------  ------
  0        0         576.22    453.42   13.94    9.48   26.32    0.14
100      200       21203.24  23940.32   88.89   88.31   89.47    0.89
200      400           0.00    316.27   90.07   90.67   89.47    0.90
300      600           1.58    308.27   88.74   89.33   88.16    0.89
400      800          30.78    338.28   91.39   92.00   90.79    0.91
500     1000           6.35    292.41   92.21   91.03   93.42    0.92
600     1200           1.00    276.68   92.21   91.03   93.42    0.92
700     1400           3.93    272.53   90.79   90.79   90.79    0.91
800     1600           0.00    249.77   90.79   90.79   90.79    0.91
900     1800           0.00    238.44   90.79   90.79   90.79    0.91
1000    2000           0.00    223.54   90.79   90.79   90.79    0.91
1100    2200           0.00    208.17   90.79   90.79   90.79    0.91
1200    2400          21.46    217.55   91.50   90.91   92.11    0.92
1300    2600           2.49    176.41   93.42   93.42   93.42    0.93
1400    2800           1.16    156.54   91.50   90.91   92.11    0.92
1500    3000           0.00    136.26   90.20   89.61   90.79    0.90
1600    3200           0.00    117.72   91.50   90.91   92.11    0.92
1700    3400           0.00     99.77   92.11   92.11   92.11    0.92
1800    3600           0.00     81.52   91.50   90.91   92.11    0.92
1900    3800           0.00     65.55   91.50   90.91   92.11    0.92
2000    4000           0.00     50.48   91.50   90.91   92.11    0.92
2100    4200           0.00     37.89   91.50   90.91   92.11    0.92
```

### Deployment
I used Gradio as my deployment solution. Other options such as Streamlit and Flask are great to explore. One could not only design web app, but also incorporate the functionality into mobile apps or back-end services. This project takes the approach of building a lightweight web app.

If you want to run the gradio locally, you will need at least the following dependencies:
```
spacy
spacy-transformers
spacy-huggingface-hub
pytorch
gradio
```

In addition, you need a spaCy model. It can be a blank one or a pre-trained one like `en_core_web_md`. This project will also demonstrate on training and fine-tuning your own spaCy model. You can find my version on [HuggingFace](https://huggingface.co/hjianganthony/en_fetch_ner_spacy_tsf?library=true). 

Next, you can run the [gradio_deploy.py](https://github.com/hjiangAnthony/retailer_nlp_challenger/blob/main/gradio_deploy.py).
