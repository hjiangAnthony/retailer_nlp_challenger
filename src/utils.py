# helper functions
from typing import List, Dict, Tuple
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

from IPython.display import clear_output

import spacy
from spacy.tokens import DocBin
from spacy.training import offsets_to_biluo_tags
import en_fetch_ner_spacy_tsf
nlp = en_fetch_ner_spacy_tsf.load()
clear_output()

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
additional_stop_words = {'pack'}
stop_words.update(additional_stop_words)
clear_output()


def single_text_cleaner(text: str, remove_stopwords: bool=False, upper_case: bool = False, remove_punctuation: bool=True) -> str:
    """Clean one single text input. By default it will convert text to lower case"""
    if upper_case:
        text = text.upper()
    else:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^a-z\s]', '', text)
    if remove_stopwords:
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    return text

def list_text_cleaner(texts: List[str], upper_case: bool = False, remove_stopwords: bool = False, remove_punctuation: bool=True) -> List[str]:
    """Takes in a list of strings and returns a list of cleaned strings without stop words. 
    Current tasks: 
    - remove non-alphabetical characters
    - converting to lower cases
    - remove stop words (optional)"""
    cleaned_texts = [single_text_cleaner(text, remove_stopwords, upper_case, remove_punctuation) for text in texts]
    return cleaned_texts

def match_product_category(s1: list[str], s2: list[str]) -> str:
    """Find if items of a list is in one list of product categories"""
    return next((p for c in s1 for p in s2 if c in p), None) # this will stop after finding first match, which saves time

def find_category(search_input: str, search_dict: Dict) -> str:
    """Find the category of a search input based on a dictionary of categories
    Args:
    - search_input: a string
    - search_dict: a dictionary of product categories
    """
    search_list = list_text_cleaner(re.split(r'[,\s]+', search_input), remove_stopwords=True)
    search_list = [c for c in search_list if len(c)>0] # sometimes there are empty strings
    matched_category = False
    for k, v in search_dict.items():
        v = list_text_cleaner(v, remove_punctuation=False)
        search_results = match_product_category(search_list, v)
        if search_results is not None:
            matched_category = True
            return k, search_results
        else:
            # print(f'Function find_category: No category {k} has matched for input: {search_input}') 
            continue
    if not matched_category:
        print(f'Function find_category: No category has matched for input: {search_input}')
        return None
    

def check_entity(search_input) -> bool:
    """Takes in a search input and checks if it contains any entities"""
    doc = nlp(search_input)
    if len(doc.ents) > 0:
        return doc
    else:
        return False

def get_cosine_sim(input_text: str, texts: List[str]) -> pd.DataFrame:
    """Calculate the cosine similarity of the input text against a list of texts
    Takes in:
    - input_text: a string
    - texts: a list of strings
    Returns a dataframe with two columns: Sentence Text and Cosine Similarity Score
    """
    input_text_cleaned = list_text_cleaner([input_text], remove_stopwords=True)[0]
    cleaned_texts = list_text_cleaner(texts, remove_stopwords=True)
    all_texts = [input_text_cleaned] + cleaned_texts
    vectors = get_vectors(*all_texts)
    sim_matrix = cosine_similarity(vectors)
    # Get the similarity scores of the input_text against all other texts
    sim_scores = sim_matrix[0, 1:]
    data = {'OFFER': texts, 'Cosine Similarity Score': sim_scores}
    df = pd.DataFrame(data)
    df = df.sort_values(by='Cosine Similarity Score', ascending=False).reset_index(drop=True)
    return df

def get_vectors(*strs: str) -> np.ndarray:
    text = list(strs)
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def jaccard_similarity(s1: List[str], s2: List[str]) -> float:
    """Takes in two lists and returns the Jaccard similarity score (3 digits)"""
    intersection = set(s1).intersection(set(s2))
    n = len(intersection)
    score = round(n / (len(s1) + len(s2) - n), 3)
    return score

def get_jaccard_sim(input_text: str, texts: List[str]) -> pd.DataFrame:
    """Calculate the Jaccard similarity of the input text against a list of texts
    Takes in:
    - input_text: a string
    - texts: a list of strings
    Returns a dataframe with two columns: Sentence Text and Jaccard Similarity Score
    """
    cleaned_input_text = list_text_cleaner([input_text], remove_stopwords=True)[0].split()
    cleaned_texts = list_text_cleaner(texts, remove_stopwords=True)
    
    jaccard_scores = [jaccard_similarity(cleaned_input_text, text.split()) for text in cleaned_texts]
    
    data = {'OFFER': texts, 'Jaccard Similarity Score': jaccard_scores}
    df = pd.DataFrame(data)
    # sort based on the similarity score
    df = df.sort_values(by='Jaccard Similarity Score', ascending=False).reset_index(drop=True)
    return df

def find_column(df: pd.DataFrame, keyword: str) -> str:
    """Function to find the first column containing a specific keyword. Note that we assume there will only be one score at most for a similarity score dataframe"""
    cols = [col for col in df.columns if keyword.lower() in col.lower()]
    return cols[0] if cols else None

def extract_similar_offers(data: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Takes in the results from get_cosine_sim() and get_jaccard_sim(); returns a dataframe of similar offers with scores > threshold"""
    score = find_column(data, 'score')
    similar_offers = data[data[score] >= threshold]
    similar_offers[score] = similar_offers[score].apply(lambda x: round(x, 3)) # round to 3 digits
    return similar_offers

def category_to_brand(category: str, offered_brands: List, brand_belong_category_dict: Dict) -> List[str]:
    """Use case: when a user searches for a category, we return a list of brands in that category"""
    # checks if the category is in the dictionary keys
    if category.upper() in brand_belong_category_dict.keys():
        search_brands = brand_belong_category_dict[category.upper()] # becase all keys are in upper case
        result = list(set(search_brands) & set(offered_brands))
        print(f"Function category_to_brand | Found {category} in offered brand") 
        return result 
    else:
        print(f"Function category_to_brand | No offered brand is found in {category}")
        return None

class CatchErros(Exception):
    class ParamsInputError(Exception):
        pass
    class SearchFailedError(Exception):
        pass
    class UnknownError(Exception):
        pass


def offer_finder_by_category(search_input: str, search_category_tuple: Tuple, category_dict: Dict, offered_brands: List, 
                             brand_belong_category_dict: Dict, score: str, threshold: float = 0.0) -> pd.DataFrame:
    """Find offers based on a category identified from search input.
    Args:
    - search_input: a string
    - search_category_tuple: a tuple of (upper_category, product_category)
    - category_dict: a dictionary of categories. Keys are upper categories and values are lists of product categories
    - offered_brands:  a dataframe of offers (OFFER, BRAND, RETAILER) that are avaialble in our database
    - brand_belong_category_dict: a dictionary of brands and the categories they belong to
    - score: a string of either 'cosine' or 'jaccard'
    - threshold: a float between 0 and 1

    Returns a dataframe of similar offers, ordered by highest score
    """
    # we assume people just search one category at a time
    # search_category_tuple = find_category(search_input, category_dict)
    product_category, upper_category = search_category_tuple[1], search_category_tuple[0] # ('Alcohol', 'beer')
    print(f'Function offer_finder_by_category | Found items:\n- Search input: {search_input}\n- Product category: {product_category}\n- Upper category: {upper_category}')
    potential_brands = category_to_brand(product_category, offered_brands, brand_belong_category_dict)
    if potential_brands is not None:
        potential_offers = offers[offers['BRAND'].isin(potential_brands)]['OFFER'].tolist()
        if score == 'cosine':
            cos_sim_score = get_cosine_sim(search_input, potential_offers)
            output = extract_similar_offers(cos_sim_score, threshold)
        elif score == 'jaccard':
            jaccard_sim_score = get_jaccard_sim(search_input, potential_offers)
            output = extract_similar_offers(jaccard_sim_score, threshold)
        elif score not in ['cosine', 'jaccard']:
            raise ValueError(f'Please enter a valid score: cosine or jaccard; Not {score}')
        else: # this means something else is worng
            raise UnknownError(f'Something must be broken. Please try again.')
        return output
    else:
        potential_product_categories = category_dict[upper_category]
        msg = f'{product_category} is not found. Do you wanna take a look at these similar offers in {upper_category}?\n We have: {potential_product_categories}' # we can still calculate similarity but this is computationally expensive
        print(msg)
        return None

def offer_finder_by_entity(search_input: str, entities: Tuple, offers_data: pd.DataFrame, score: str, threshold: float=0.0) -> pd.DataFrame:
    """Find offers based on entities identified from search input.
    Args:
    - search_input: a string
    - entities: a tuple of entities
    - offers_data: a dataframe of offers (OFFER, BRAND, RETAILER) that are avaialble in our database
    - score: a string of either 'cosine' or 'jaccard'
    - threshold: a float between 0 and 1

    Returns a dataframe of similar offers, ordered by highest score
    """
    collects = [] # collect all the results if there are more than one entity
    for ent in entities:
        ent_name, ent_label = ent.text, ent.label_
        print(f'Function offer_finder_by_entity | Found entity: {ent_name} with label: {ent_label}')
        # filter offers by entity
        df_tmp = offers_data[offers_data[ent_label.upper()] == ent_name.upper()]
        if df_tmp.shape[0] > 0:
            print(f'Function offer_finder_by_entity | Found {df_tmp.shape[0]} offer(s) for the brand/retailer: {ent_name}')
            potential_offers = df_tmp['OFFER'].drop_duplicates().tolist()
            if score == 'cosine':
                cos_sim_score = get_cosine_sim(search_input, potential_offers)
                output = extract_similar_offers(cos_sim_score, threshold)
            elif score == 'jaccard':
                jaccard_sim_score = get_jaccard_sim(search_input, potential_offers)
                output = extract_similar_offers(jaccard_sim_score, threshold)
            elif score not in ['cosine', 'jaccard']:
                raise ValueError(f'Please enter a valid score: cosine or jaccard; Not {score}')
            else: # this means something else is worng
                raise UnknownError(f'Something must be broken. Please try again.')
            collects.append(output)
        else:
            print(f'Function offer_finder_by_entity | No offer is found for the brand/retailer: {ent_name}')

    if len(collects) > 0:
        final_output = pd.concat(collects, ignore_index=True)# they should be using the same similarity score
        score = find_column(collects[0], 'score') 
        final_output = final_output.sort_values(by=score, ascending=False).reset_index(drop=True) # sort final_output by score
        return final_output
    elif len(collects) == 1:
        return collects[0]
    else:
        print('###'*5 + 'FINAL SEARCH RESULTS' + '###'*5)
        print('Function offer_finder_by_entity | No offer is found for any of the entities.')
        return None


def main(search_input: str, offers: pd.DataFrame, category_dict: Dict, brand_belong_category_dict: Dict, score: str, score_threshold: float = 0.0):
    """Main function. Takes in a serach_input and decide whether it can find entities or not. Then excecute the appropriate functions
    Inputs:
    - search_input: a string that a user enters
    - offers: a dataframe of offers (OFFER, BRAND, RETAILER) that are avaialble in our database
    - category_dict: a dictionary of categories. Keys are upper categories and values are lists of product categories
    - brand_belong_category_dict: a dictionary of brands and the categories they belong to
    - score: a string of either 'cosine' or 'jaccard'
    - score_threshold: a float between 0 and 1

    Returns a dataframe of similar offers, ordered by highest score
    """
    print(f'Function main | Search input: {search_input}')
    check_ent = check_entity(search_input)
    if not check_entity(search_input): # no entities found
       # check category
       cat_check = find_category(search_input, category_dict)
       if cat_check is None:
           raise SearchFailedError('No brand/retailer/category is found. Please try again.')
       else:
            # we assume people just search one category at a time
            cat_tuple = cat_check # ('Alcohol', 'beer')
            search_results = offer_finder_by_category(search_input, cat_tuple, category_dict, offers, brand_belong_category_dict, score, score_threshold)
            return search_results
    else:
        entities = check_ent.ents # entities will be a tuple anyways
        print(f'Found {len(entities)} entity object(s) in the search input.')
        search_results = offer_finder_by_entity(search_input, entities, offers, score, score_threshold)
        if search_results is None:
            print('No offers matched retailer/category is found. Now trying to recommend based on category.')
            cat_check = find_category(search_input, category_dict)
            if cat_check is None:
                raise SearchFailedError('No brand/retailer/category is found. Please try again.')
            else:
                cat_tuple = cat_check
                search_results = offer_finder_by_category(search_input, cat_tuple, category_dict, offers, brand_belong_category_dict, score, score_threshold)
        return search_results
            