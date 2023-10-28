import gradio as gr
import pandas as pd
import numpy as np

from src.utils import *

##### Start #####

examples = [
    ["Simply Spiked Lemonade 12 pack at Walmart", "jaccard", 0.1, 0.1],
    ["Back to the Roots Garden Soil, 1 cubic foot, at Lowe's Home Improvement", "jaccard", 0.1, 0.1],
    ["Costco Member subscription", "jaccard", 0.1, 0.1],
    ["Apple watch coupon at Best Buy", "jaccard", 0.1, 0.1],
    ["A giraffe at Lincoln Park Zoo", "jaccard", 0.1, 0.1]
]

def main(sentence: str, score_type: str, threshold_cosine: float, threshold_jaccard: float = 0.1):
    threshold = threshold_cosine if score_type == "cosine" else threshold_jaccard    
    results = search_offers(search_input=sentence, 
                            score=score_type, 
                            score_threshold=threshold)
    message, processed_results = process_output(results)
    return message, processed_results

def process_output(output):
    """Function to process the output"""
    if output is None or output.empty:
        return "We couldn't find your results, please try our examples or search again", None
    else:
        return "We found some great offers!", output

demo = gr.Interface(
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
    demo.launch(share=True)