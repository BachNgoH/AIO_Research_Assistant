from llama_index.core.text_splitter import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from tqdm import tqdm
import json
from datasets import load_dataset
from utils.helper import has_citation
from utils.prompts import DEFAULT_CITATION_INFER_PROMPT_TEMPLATE as prompt_template
from unsloth import FastLanguageModel
import random
random.seed(42)
import argparse

LLM_SERVICE = "hf"

sentence_splitter = SentenceSplitter.from_defaults(chunk_size=256, chunk_overlap=0)

if LLM_SERVICE == "openai":
    llm = OpenAI(model="gpt-3.5-turbo", api_key="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")
elif LLM_SERVICE == "hf":
    max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "../output/Gemma_7b_Citation_v2", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        token = "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", # use one if using gated models like meta-llama/Llama-2-7b-hf
        # cache_dir = "../output/models",
    )
    FastLanguageModel.for_inference(model) 

    llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)

def main(args):
    generated_data = []
    all_articles = []
    
    # with open(f'../output/generated_data_v4.json', 'r') as f:
    #     generated_data = json.load(f)
    all_articles = load_dataset(args.dataset_name)['train']
    all_articles = all_articles.to_list()

    print("TOTAL len of all articles: ",  len(all_articles))

    for article in tqdm(all_articles[400:800], total=len(all_articles[400:800])):
        chunks = []
        article['citation_data'] = []
        for section in article['sections']:
            if len(section['publication_ref']) > 0:
                res = sentence_splitter.split_text(section['text'])
                # chunks += res
                chunks += [c for c in res if has_citation(c)]
        for chunk in chunks:
            try:
                completion = llm.complete(prompt_template.format(input=chunk))
                # citation_data = parse_json(completion.text)
                citation_data = completion.text
                generated_data.append({'Input': chunk, 'Output': citation_data})

                article['citation_data'] += citation_data
            except Exception as e:
                print('Error in chunk: {e}')
                continue
        # save generated_data
        with open(f'{args.output_path}/generated_data_v3.json', 'w') as f:
            json.dump(generated_data, f)

        # save new article data
        with open(f'{args.output_path}/parsed_article_w_citation_v2.json', 'w') as f:
            json.dump(all_articles, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BachNgoHParsedArxivPapers_12k')
    parser.add_argument('--output_path', type=str, default='../outputs')
    args = parser.parse_args()
    main(args)
            
