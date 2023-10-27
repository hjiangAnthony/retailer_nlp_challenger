import gradio as gr
import pandas as pd
import numpy as np

import pickle, json
from src.utils import *

##### Start #####

# load operation data
path1 = "data/brand_belong_category_dict.json"
path2 = "data/product_upper_category_dict.json"
path3 = "data/offered_brands.pkl"
path4 = "data/offer_retailer.csv"

with open(path1, 'r') as f:
    brand_belong_category_dict = json.load(f)

with open(path2, 'rb') as f:
    category_dict = json.load(f)

with open(path3, 'rb') as f:
    offered_brands = pickle.load(f)

df_offers_brand_retailer = pd.read_csv(path4)

examples = [
    ["Simply Spiked Lemonade 12 pack at Walmart"],
    ["Back to the Roots Garden Soil, 1 cubic foot, at Lowe's Home Improvement"],
    ["Costco Member subscription"],
    ["Apple watch coupon at Best Buy"],
    ["A giraffe at Lincoln Park Zoo"]
]

def main(sentence: str, score_type: str, threshold_cosine: float, threshold_jaccard: float):
    threshold = threshold_cosine if score_type == "cosine" else threshold_jaccard    
    results = search_offers(sentence, df_offers_brand_retailer, category_dict, brand_belong_category_dict, score_type, threshold)
    message, processed_results = process_output(results)
    return message, processed_results

def process_output(output):
    """Function to process the output"""
    if output is None or output.empty:
        return "We couldn't find your results, please try our examples or search again", None
    else:
        return "We found some great offers!", output

iface = gr.Interface(
    fn=main,
    inputs=[
        gr.Textbox(lines=1, placeholder="Type here..."),
        gr.Dropdown(choices=["cosine", "jaccard"], label="Score Type"),
        gr.Slider(minimum=0, maximum=1, step=0.1, label="Threshold for Cosine Similarity"),
        gr.Slider(minimum=0, maximum=1, step=0.1, label="Threshold for Jaccard Similarity")
    ],
    outputs=[gr.Textbox(placeholder="Message..."), gr.Dataframe()],
    examples=examples,
    live=False,
)

if __name__ == "__main__":
    iface.launch()