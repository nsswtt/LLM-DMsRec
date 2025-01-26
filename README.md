![image text](https://github.com/nsswtt/LLM-DMsRec/blob/main/dataset/model.png "LLM-DMsRec")
# Integrating LLM-Derived Multi-Semantic Intent into Graph Model for Session-based Recommendation
## abstract
Session-based recommendation (SBR) is mainly based on anonymous user interaction sequences to recommend the items that the next user is most likely to click.  Currently, the most popular and high-performing SBR methods primarily leverage graph neural networks (GNNs), which model session sequences as graph-structured data to effectively capture user intent. However, most GNNs-based SBR methods primarily focus on modeling the ID sequence information of session sequences, while neglecting the rich semantic information embedded within them. This limitation significantly hampers model's ability to accurately infer users' true intention. To address above challenge, this paper proposes a novel SBR approach called Integrating LLM-Derived Multi-Semantic Intent into Graph Model for Session-based Recommendation ( LLM-DMsRec). The method utilizes a pre-trained GNN model to select the top-k items as candidate item sets and designs prompts along with a large language model (LLM) to infer multi-semantic intents from these candidate items. Specifically, we propose an alignment mechanism that effectively integrates the semantic intent inferred by the LLM with the structural intent captured by GNNs. Extensive experiments conducted on the Beauty and ML-1M datasets demonstrate that the proposed method can be seamlessly integrated into GNNs framework, significantly enhancing its recommendation performance.

## dataset
### beauty(2014)
 https://jmcauley.ucsd.edu/data/amazon/links.html
### ml-1m
 https://grouplens.org/datasets/movielens/

## LLM(Qwen2.5-7B)
 https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

## train
1. pre-train GNN obtains candidate item sets
3. LLM infer user intentions  
   `cd dataset`  
   `main infer_intent.py`
5. GNN trainer  
   `cd src`  
   `main main_srgnn.py`  
   `main main_tagnn.py`  
   `main main_gcegnn.py`  
   `main main_share.py`
   `main main_msgifsr.py`
## note
The file path needs to be set by yourself.
