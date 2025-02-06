import pandas as pd
import zipfile
from datasets import load_dataset, Dataset
from typing import Callable, List, Union, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
import sys
import json
# from sentence_transformers import SentenceTransformer
# from faiss import IndexFlatIP
import random
  



######################### This block for the data loading #####################################################
def load_qa(**kwargs) -> Dataset:
    return load_dataset(
        "cardiffnlp/databench", **{"name": "qa", "split": "train", **kwargs}
    )

def load_table(name):
    return pd.read_parquet(
        f"/home/yuze/Workspace/semeval/data/databench/data/{name}/all.parquet"
    )


def load_sample(name):
    return pd.read_parquet(
        f"/home/yuze/Workspace/semeval/data/databench/data/{name}/sample.parquet"
    )
########################### End of the data loading #########################################################




 
############### Define the global variables #####################################################################
class Config:
    batch_size = 10
    similiar_shot_number = 8 # number of similiar shots to retrieve
    database_sample_number = 5 # number of database samples to retrieve
    model_card = "meta-llama/Llama-3.3-70B-Instruct"
    features = ['col_names', 'similiar_shots', 'col_types', 'row_samples']
    re_gen_function = True
global_config = Config()
print(sys.argv)
global_config.model_card = sys.argv[1] if len(sys.argv) > 1 else global_config.model_card
################### End of the global variables ################################################################








################### This Block to load large LLM ############################
# please Use CUDA_VISIBLE_DEVICES to select the GPUs to use, 70B model needs at least 4 A6000 GPUs for inference
model_card = global_config.model_card
if model_card == "meta-llama/Llama-3.3-70B-Instruct":
    weights_path = snapshot_download(model_card)
    model_card = weights_path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_card, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_card, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = load_checkpoint_and_dispatch(model, model_card,
                                        device_map='auto',
                                        offload_folder="offload",
                                        offload_state_dict=True,
                                        dtype = "float16",
                                        no_split_module_classes=["LlamaDecoderLayer"])

elif model_card == "meta-llama/Llama-3.1-8B-Instruct":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_card)
    # config = AutoConfig.from_pretrained(model_card, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_card).cuda()

elif model_card == "meta-llama/Llama-3.3-70B-Instruct-v3":
    pass
##################### End of the loading #################################










#################### This Block to call the loaded LLM ############################
def call_original_llama_model(prompts):
    results = []
    for p in prompts:
        print(f"In call_original_llama_model, prompt is: {p}")
        p = p.replace('"', '\\"')
        inputs = tokenizer(p, return_tensors="pt").to("cuda:0")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = model.generate(
                    input_ids=input_ids,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    min_length=5,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=60
                )
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print("@@@@@@@@@@@@@@@@@ In call_original_llama_model, prompt is: ", p)
        # print("@@@@@@@@@@@@@@@@@ In call_original_llama_model, result is: ", result[0])
        results.append(result[0])
    return results
################### End of the calling #################################












########################### This block for the prompt generation##################################################
def example_generator(row) -> str:
    dataset = row["dataset"]
    question = row["question"]
    return_type = row["type"]
    print(f"In example_generator, dataset is: {dataset}, question is: {question}, return_type is: {return_type}")
    df = load_table(dataset)
    if ('similiar_shots' in global_config.features):
        shots = retrieve_similiar_shots_by_type(question, return_typ=return_type)
        assert len(shots) == global_config.similiar_shot_number, f"Expected {global_config.similiar_shot_number} similiar shots, got {len(shots)}"
        similiar_shot_content = ""
        for sid, shot in enumerate(shots):
            similiar_shot_content += f"""
# example {5+sid}, similiar case
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: {shot['question']}
def answer(df: pd.DataFrame) -> category:
    df.columns = {shot['columns']}
    df.column_types = {str(shot['column_types'])}
    return {shot['df_func']} ############

"""
    prompt = """
# Instruction: You are proficient in pandas and its functions to retrieve data from a dataframe. Please complete the following function in one line. Be careful with the case, whitespaces and special characters in the column name. End your answer with ############

# example 1
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: How many rows are there in this dataframe? 
def answer(df: pd.DataFrame) -> number:
    df.columns=["A"]
    return df.shape[0] ############
# example 2
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: What are the column names of this dataframe? 
def answer(df: pd.DataFrame) -> list[category]:
    return df.columns.tolist() ############

# example 3, complex level
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: List the top 5 ranks of billionaires who are not self-made.
def answer(df: pd.DataFrame) -> list[number]:
    df.columns = 'rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
    return df.loc[df['selfMade'] == False].head(5)['rank'].tolist() ############

# example 4, complex level
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: Which category does the richest billionaire belong to?
def answer(df: pd.DataFrame) -> category:
    df.columns = ['rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
    return df.loc[df['finalWorth'].idxmax()]['category'] ############
""" 
    if ('similiar_shots' in global_config.features):
        prompt += similiar_shot_content


    prompt += f"""

# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: {question}
def answer(df: pd.DataFrame) -> {row["type"]}:
    df.columns = {list(df.columns)} # column names
"""

    if 'col_types' in global_config.features:
        prompt += f"""
    df.column_types = {str([itm.name for itm in df.dtypes])} # column types
"""
    
    if 'row_samples' in global_config.features:
        prompt += f"""
    first{global_config.database_sample_number}_row_samples = {df.head(global_config.database_sample_number).to_dict(orient='records')}"""
    prompt += """
    return"""
    return prompt
########################### End of the prompt generation ############################################################

#################### This block for the prompt generation lite ##################################################
def example_generator_lite(row) -> str:
    dataset = row["dataset"]
    question = row["question"]
    return_type = row["type"]
    print(f"In example_generator, dataset is: {dataset}, question is: {question}, return_type is: {return_type}")
    df = load_sample(dataset)
    if ('similiar_shots' in global_config.features):
        shots = retrieve_similiar_shots_by_type(question, return_typ=return_type)
        assert len(shots) == global_config.similiar_shot_number, f"Expected {global_config.similiar_shot_number} similiar shots, got {len(shots)}"
        similiar_shot_content = ""
        for sid, shot in enumerate(shots):
            similiar_shot_content += f"""
# example {5+sid}, similiar case
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: {shot['question']}
def answer(df: pd.DataFrame) -> category:
    df.columns = {shot['columns']}
    df.column_types = {str(shot['column_types'])}
    return {shot['df_func']} ############

"""
    prompt = """
# Instruction: You are proficient in pandas and its functions to retrieve data from a dataframe. Please complete the following function in one line. Be careful with the case, whitespaces and special characters in the column name. End your answer with ############

# example 1
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: How many rows are there in this dataframe? 
def answer(df: pd.DataFrame) -> number:
    df.columns=["A"]
    return df.shape[0] ############
# example 2
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: What are the column names of this dataframe? 
def answer(df: pd.DataFrame) -> list[category]:
    return df.columns.tolist() ############

# example 3, complex level
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: List the top 5 ranks of billionaires who are not self-made.
def answer(df: pd.DataFrame) -> list[number]:
    df.columns = 'rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
    return df.loc[df['selfMade'] == False].head(5)['rank'].tolist() ############

# example 4, complex level
# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: Which category does the richest billionaire belong to?
def answer(df: pd.DataFrame) -> category:
    df.columns = ['rank', 'personName', 'age', 'finalWorth', 'category', 'source', 'country', 'state', 'city', 'organization', 'selfMade', 'gender', 'birthDate', 'title', 'philanthropyScore', 'bio', 'about']
    return df.loc[df['finalWorth'].idxmax()]['category'] ############
""" 
    if ('similiar_shots' in global_config.features):
        prompt += similiar_shot_content


    prompt += f"""

# TODO: complete the following function in one line, response type in ['boolean', 'category', 'list[category]', 'list[number]', 'number']. It should give the answer to: {question}
def answer(df: pd.DataFrame) -> {row["type"]}:
    df.columns = {list(df.columns)} # column names
"""

    if 'col_types' in global_config.features:
        prompt += f"""
    df.column_types = {str([itm.name for itm in df.dtypes])} # column types
"""
    
    if 'row_samples' in global_config.features:
        prompt += f"""
    first{global_config.database_sample_number}_row_samples = {df.head(global_config.database_sample_number).to_dict(orient='records')}"""
    prompt += """
    return"""
    return prompt
###################### End of the prompt generation ##########







#################### prompt to get answer type ########################################
def get_answer_type_prompt(row: dict) -> str:
    dataset = row["dataset"]
    question = row["question"]
    df = load_table(dataset)
    return f"""
# Instruction: You are proficient in database and can easily tell the type of the retrieving answer for the question. Please complete the following function in one line. End your answer with ############
answer_types = ['boolean', 'category', 'list[category]', 'list[number]', 'number']
# TODO: complete the following function in one line. It should give the answer type for: List the 3 patents (by ID) with the most number of claims. 
def get_answer_type() -> str:
    answer_types = ['boolean', 'category', 'list[category]', 'list[number]', 'number']
    question = "List the 3 patents (by ID) with the most number of claims."
    return 'list[number]' ############

# TODO: complete the following function in one line. It should give the answer type for: Which graphext cluster is the most common among the patents? 
def get_answer_type() -> str:
    answer_types = ['boolean', 'category', 'list[category]', 'list[number]', 'number']
    question = "Which graphext cluster is the most common among the patents?"
    return 'category' ############

# TODO: complete the following function in one line. It should give the answer type for: List the 2 most common types of patents in the dataset. 
def get_answer_type() -> str:
    answer_types = ['boolean', 'category', 'list[category]', 'list[number]', 'number']
    question = "List the 2 most common types of patents in the dataset."
    return 'list[category]' ############

# TODO: complete the following function in one line. It should give the answer type for: Is the most favorited author mainly communicating in Spanish?. 
def get_answer_type() -> str:
    answer_types = ['boolean', 'category', 'list[category]', 'list[number]', 'number']
    question = "Is the most favorited author mainly communicating in Spanish?"
    return 'list[category]' ############

# TODO: complete the following function in one line. It should give the answer type for: {question}
def answer() -> str:
    answer_types = ['boolean', 'category', 'list[category]', 'list[number]', 'number']
    question = "{question}"
    return"""

############################# End of the answer type ###################################################################











################### re-gen based on error ########################################
def get_a_new_function(row: dict, columns: List[str], datafrm: pd.DataFrame, error_function: str, error_message: str):
    question = row["question"]
    # write a prompt to ask the LLM to rewrite the function based the columns, the old function and error message
    # return the new function
    columns = str(columns)
    column_types = str([itm.name for itm in datafrm.dtypes])
    prompt = f"""
Instruction: You are proficient in pandas and its functions to retrieve data from a dataframe. Please complete the following function in one line. End your answer with ############
# example 1
# Todo: Rewrite the pandas function based on the columns, the old function and the error message. It should give the right pandas function to: What is the average unit price? 
def check_and_fix_function(question: str, columns: List[str], error_function: str, error_message: str) -> str:
    question = "What is the average unit price?"
    columns = ['InvoiceNo', 'Country', 'StockCode', 'Description', 'Quantity', 'CustomerID', 'UnitPrice']
    error_function =  df[' UnitPrice'].mean()
    error_message = ' UnitPrice' # unexpected whitespace
    return  df['UnitPrice'].mean()  ############

# example 2
# Todo: Rewrite the pandas function based on the columns, the old function and the error message. It should give the right pandas function to:  What is the most commonly achieved educational level among the respondents? 
def check_and_fix_function(question: str, columns: List[str], error_function: str, error_message: str) -> str:
    question = " What is the most commonly achieved educational level among the respondents?"
    columns = ['Are you registered to vote?', 'Which of the following best describes your ethnic heritage?', 'Who are you most likely to vote for on election day?', 'Division', 'Did you vote in the 2016 Presidential election? (Four years ago)', 'Weight', 'How likely are you to vote in the forthcoming US Presidential election? Early Voting Open', 'State', 'County FIPS', 'Who did you vote for in the 2016 Presidential election? (Four years ago)', 'What is the highest degree or level of school you have *completed* ?', 'NCHS Urban/rural', 'likelihood', 'Which of these best describes the kind of work you do?', 'How old are you?']
    error_function = df['What is the highest degree or level of school you have *completed*?'].value_counts().idxmax()
    error_message = "What is the highest degree or level of school you have *completed*?" # missed a whitespace
    return df['What is the highest degree or level of school you have *completed* ?'].value_counts().idxmax() ############

# example 3
# Todo: Rewrite the pandas function based on the columns, the old function and the error message. It should give the right pandas function to: Who are the top 2 authors of the tweets with the most retweets? 
def check_and_fix_function(question: str, columns: List[str], error_function: str, error_message: str) -> str:
    question = "Who are the top 2 authors of the tweets with the most retweets?"
    columns = ['id<gx:category>', 'author_id<gx:category>', 'author_name<gx:category>', 'author_handler<gx:category>', 'author_avatar<gx:url>', 'user_created_at<gx:date>', 'user_description<gx:text>', 'user_favourites_count<gx:number>', 'user_followers_count<gx:number>', 'user_following_count<gx:number>', 'user_listed_count<gx:number>', 'user_tweets_count<gx:number>', 'user_verified<gx:boolean>', 'user_location<gx:text>', 'lang<gx:category>', 'type<gx:category>', 'text<gx:text>', 'date<gx:date>', 'mention_ids<gx:list[category]>', 'mention_names<gx:list[category]>', 'retweets<gx:number>', 'favorites<gx:number>', 'replies<gx:number>', 'quotes<gx:number>', 'links<gx:list[url]>', 'links_first<gx:url>', 'image_links<gx:list[url]>', 'image_links_first<gx:url>', 'rp_user_id<gx:category>', 'rp_user_name<gx:category>', 'location<gx:text>', 'tweet_link<gx:url>', 'source<gx:text>', 'search<gx:category>']
    error_function = df.nlargest(2,'retweets')['author_name<gx:category>'].tolist()
    error_message = 'retweets'
    return df.nlargest(2, 'retweets<gx:number>')['author_name<gx:category>'].tolist() ############

# example 4
# Todo: Rewrite the pandas function based on the columns, the old function and the error message. It should give the right pandas function to: Is there a patent abstract that mentions 'software'? 
def check_and_fix_function(question: str, columns: List[str], error_function: str, error_message: str) -> str:
    question = "Is there a patent abstract that mentions 'software'?"
    columns = ['kind', 'num_claims', 'title', 'date', 'lang', 'id', 'abstract', 'type', 'target', 'graphext_cluster', 'organization']
    error_function = ('software' in df['abstract'].values).any()
    error_message = "'bool' object has no attribute 'any'"
    return ('software' in df['abstract'].values) ############

# Todo: Rewrite the pandas function based on the columns, the old function and the error message. It should give the right pandas function to: {question} 
def check_and_fix_function(question: str, columns: List[str], error_function: str, error_message: str) -> str:
    question =  {question}
    columns =   {columns}
    columns_types = {column_types}
    error_function =  {error_function}
    error_message =  {error_message}
    return  
"""
    new_func = call_original_llama_model([prompt])[0]
    new_func = new_func.split("return")[5].strip()
    if "#" in new_func:
        new_func = new_func.strip().split('#')[0]
    if 'def ' in new_func:
        new_func = new_func.split("def ")[0]
    return new_func
############################### End of re-gen based on error ############################################################





################### Example postprocess ########################################
################### 1. main function ##########
def example_postprocess(response: str, dataset: str, row, loader):
    split_number = 5+global_config.similiar_shot_number if ('similiar_shots' in global_config.features) else 5
    print(f"In example_postprocess, response is: \n\n\n{response}\n\n\n")
    print(f"split_number is: {split_number}")
    response = response.split("return")[split_number]
    if "#" in response:
        response = response.strip().split('#')[0]
    if 'def ' in response:
        response = response.split("def ")[0]

    df = loader(dataset)
    global ans
    try:
        ans = eval(response)
    except Exception as e:
        e = str(e).replace('\n', ' ')
        if global_config.re_gen_function:
            try:
                response = get_a_new_function(row=row, columns=df.columns.tolist(), datafrm=df, error_function=response, error_message=e)
                ans = eval(response)
            except Exception as e:
                e = str(e).replace('\n', ' ')
                col = match_string_with_column_name_list(e, df.columns.tolist())
                if col in df.columns.tolist():
                    response = response.replace(e, "'"+col+"'").strip()
                try:
                    ans = eval(response)
                except Exception as e:
                    e = str(e).replace('\n', ' ')
                    ans = 'Wrong Code'
                    response = response.replace("\n", " ")
                    print(f"__CODE_ERROR__: {e} in response {response} for answer {ans}, #### Columns are: {str(df.columns.tolist())}, ### question is: {row['question']}")
                    return f"__CODE_ERROR__: {e} in respones {response} for answer {ans}, #### Columns are: {str(df.columns.tolist())}, ### question is: {row['question']}"
        else:
            print(f"__CODE_ERROR__: {e} in response {response} for answer {ans}, #### Columns are: {str(df.columns.tolist())}, ### question is: {row['question']}")
            return f"__CODE_ERROR__: {e} in respones {response} for answer {ans}, #### Columns are: {str(df.columns.tolist())}, ### question is: {row['question']}"
    ans = post_process_ans_return(ans)    
    ans = ans.split("\n")[0] if "\n" in str(ans) else ans
    return ans
######################### End of main function ###########################################################################

################### 2. post_process_ans_return ##########
def post_process_ans_return(response):
    """
    Post-process the model's answer into a string representation.
    Handles lists, scalars, pandas Series/DataFrame, and categorical data.

    - Lists are converted to their string representation.
    - Scalars are converted to strings.
    - Pandas Series and DataFrames are converted by turning their elements into strings
      and then using `.to_string()` to produce a readable result.
    - Categorical data is handled by converting each element to a string.
    """
    
    # If response is None or already a string/scalar (int, float, bool, etc.), just return str
    if response is None or isinstance(response, (int, float, bool, str)):
        return str(response)
    
    # If response is a list, convert it to string
    if isinstance(response, list):
        return str(response)
    # If it's a Pandas Series
    if isinstance(response, pd.Series):
        # Convert categorical or object dtype elements to string individually
        # response = response.
        response = response.to_list()
        return str(response)
    # If it's a Pandas DataFrame
    if isinstance(response, pd.DataFrame):
        # Convert all elements to string before using to_string
        df_str = response.values.ravel().tolist()
        return str(df_str)
    # If it's some other type (e.g., numpy array or other objects), fallback to str
    return str(response)
#################### End of post_process_ans_return ######################################################

################### 3. match_string_with_column_name_list ##########
def match_string_with_column_name_list(string, column_name_list):
    # remove " or ' from string begin&end
    if string[0] == string[-1] and (string[0] == '"' or string[0] == "'"):
        string = string[1:-1]

    column_name_with_no_space = [column_name.replace(' ', '') for column_name in column_name_list]  # remove space
    string_with_no_space = string.replace(' ', '')  # remove space


    # 1. this case is to find the space problem
    col_index = -1

    try:
        col_index = column_name_with_no_space.index(string_with_no_space)
    except ValueError:
        pass

    if col_index != -1:
        return column_name_list[col_index]
    
    # 2. this case is to find the partial match
    for column_name in column_name_list:
        if string in column_name and string[0] == column_name[0]:
            col_index = column_name_list.index(column_name)
            break
    if col_index != -1:
        return column_name_list[col_index]
    return string
############################  End of Example postprocess ########################################





############################ Function to get similiar shots ########################################
######################## 1. by question and return type ############################
# sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Use a model potentially better suited for semantic similarity tasks
# sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# def load_examples(filename):
#     json_exmaples = json.load(open(filename, "r"))
#     json_exmaples_dict = {itm['question']+'\t'+itm['types']+' '+itm['types']: itm for itm in json_exmaples}
#     examples_keys = list(json_exmaples_dict.keys())
#     examples_keys_embeeded = sentence_model.encode(examples_keys)
#     return examples_keys_embeeded, json_exmaples_dict, examples_keys

# # Create FAISS index
# def create_faiss_index(examples_keys_embeeded):
#     d = examples_keys_embeeded.shape[1]
#     index = IndexFlatIP(d)
#     index.add(examples_keys_embeeded)
#     return index

# # Load data and create FAISS index (assuming total_true.json is your data file)
# examples_keys_embeeded, json_exmaples_dict, examples_keys = load_examples("/home/yuze/Workspace/semeval/methods/1.Code_generation/total_true_filterred.json")
# index = create_faiss_index(examples_keys_embeeded)

# def retrieve_similiar_shots(question: str, return_typ: str, k=3, max_dis=0.8):
#     # write a prompt to ask the LLM to retrieve the similiar shots based on the question and return type
#     # return the similiar shots
#     new_question_type = question + '\t' + return_typ + ' ' + return_typ
#     new_embedding = sentence_model.encode(new_question_type)
#     distances, indices = index.search(new_embedding.reshape(1, -1), k)

#     # Ensure k examples are always returned
#     similiar_shots = []
#     for idx in indices[0]:
#         similiar_shots.append(json_exmaples_dict[examples_keys[idx]])

#     # Randomly sample if not enough similar shots found
#     if len(similiar_shots) < k:
#         missing_shots = k - len(similiar_shots)
#         random_indices = random.sample(range(len(json_exmaples_dict)), missing_shots)
#         for idx in random_indices:
#             similiar_shots.append(list(json_exmaples_dict.values())[idx])
#     return similiar_shots

############ 2. by return type ############
json_exmaples = json.load(open("/home/yuze/Workspace/semeval/methods/1.Code_generation/total_true_filterred.json", "r"))
json_exmaples_dict = {}
for itm in json_exmaples:
    question = itm['question']
    types = itm['types']
    if types not in json_exmaples_dict:
        json_exmaples_dict[types] = []
    json_exmaples_dict[types].append(itm)

def retrieve_similiar_shots_by_type(question: str, return_typ: str, k=global_config.similiar_shot_number):
    type = return_typ
    if type not in json_exmaples_dict:
        return []
    examples = json_exmaples_dict[type]
    question_start_words = question.split()[0]
    shorted_examples = [itm for itm in examples if itm['question'].split()[0] == question_start_words]
    if len(shorted_examples) <= k:
        return shorted_examples + random.sample(examples, k-len(shorted_examples))
    else:
        return random.sample(shorted_examples, k)
#################################### End of Function to get similiar shots #################################################






############################ Class Runner ########################################
class Runner:
    def __init__(
        self,
        model_call: Callable,
        prompt_generator: Optional[Callable] = None,
        postprocess: Optional[Callable] = None,
        qa: Optional[Dataset] = None,
        batch_size: int = 10,
        **kwargs,
    ):
        self.model_call = model_call
        self.prompt_generator = prompt_generator
        if postprocess is not None:
            self.postprocess = postprocess

        self.raw_responses = []
        self.responses = []
        self.prompts = []
        self.qa: Dataset = qa if qa is not None else load_qa(**kwargs)
        self.batch_size = batch_size

    def process_prompt(self, prompts, datasets, batch_rows):
        # get the answer type for each prompt
        raw_responses = self.model_call(prompts)

        responses = [
            self.postprocess(response=raw_response, dataset=dataset, row=row)
            for raw_response, dataset, row in zip(raw_responses, datasets, batch_rows)
        ]
        
        self.prompts.extend(prompts)
        self.raw_responses.extend(raw_responses)
        self.responses.extend(responses)
    
    def get_answer_type(self, type_prompts):
        """Give a question and a dataframe, return the answer type."""
        """Answer types: boolean, category, list[category], list[number], number"""
        batched_types = self.model_call(type_prompts)
        # postprocess the answer type, split based on #, and return the answer type
        batched_types = [itm.split("return")[5].strip() for itm in batched_types]
        batched_types = [itm.split("#")[0].strip(' "\'') for itm in batched_types]

        return batched_types


    def run(
        self,
        prompts: Optional[list[str]] = None,
        save: Optional[str] = None,
    ) -> List[str]:
        if prompts is not None:
            if len(prompts) != len(self.qa):
                raise ValueError("n_prompts != n_qa")

            for i in tqdm(range(0, len(prompts), self.batch_size)):
                batch_prompts = prompts[i : i + self.batch_size]
                batch_datasets = self.qa[i : i + self.batch_size]["dataset"]
                self.process_prompt(batch_prompts, batch_datasets)
        else:
            if self.prompt_generator is None:
                raise ValueError("Generator must be provided if prompts are not.")
            for i in tqdm(range(0, len(self.qa), self.batch_size)):
                print('selecting rows, batch id = ', i)
                batch_rows = self.qa.select(
                    range(i, min(i + self.batch_size, len(self.qa)))
                )
                batch_type_prompts = [get_answer_type_prompt(row) for row in batch_rows]
                batched_types = self.get_answer_type(batch_type_prompts)
                new_batch_rows = []
                for tidx,tty in enumerate(batched_types):
                    itm = batch_rows[tidx]
                    itm.update({'type':tty})
                    new_batch_rows.append(itm)
                batch_rows = new_batch_rows
                # print('processing prompts, batch rows = ', batch_rows)
                batch_prompts = [self.prompt_generator(row) for row in batch_rows]
                # print('processing prompts, batch id = ', i)
                # print('@@@@@@@@@@@@@@', batch_prompts)
                batch_datasets = [row["dataset"] for row in batch_rows]
                self.process_prompt(batch_prompts, batch_datasets, batch_rows)

        if save is not None:
            self.save_responses(save)
        return self.responses

    def save_responses(self, save_path: str) -> None:
        with open(save_path, "w") as f:
            for response in self.responses:
                f.write(str(response) + "\n")

    def postprocess(self, response, dataset):
        return response
 ########################### End of Class Runner ####################



################## Run the code ########################################
##################################################################
qa = load_dataset("csv", data_files={'test':'./test_qa.csv'})['test']



runner = Runner(
    model_call=call_original_llama_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset, row: example_postprocess(response, dataset, row, load_table),
    qa=qa,
    batch_size=global_config.batch_size,
)

runner_lite = Runner(
    model_call=call_original_llama_model,
    prompt_generator = example_generator_lite,
    postprocess = lambda response, dataset, row: example_postprocess(response, dataset, row, load_sample),
    qa=qa,
    batch_size=global_config.batch_size,
)

responses = runner.run(save="predictions.txt")
responses_lite = runner_lite.run(save="predictions_lite.txt")


with zipfile.ZipFile("submission.zip", "w") as zipf:
    zipf.write("predictions.txt")
    zipf.write("predictions_lite.txt")

print("Created submission.zip containing predictions.txt and predictions_lite.txt")
