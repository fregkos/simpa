

from tqdm import tqdm
# from preprocessing import preprocess_data
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import re
import gc
import glob

from sentence_transformers import SentenceTransformer

# In order to classify a document based on the sentence transformer
# we need to prepend the 'classification: ' text before each piece of text
def main():
  model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
  sentences = ['classification: the quick brown fox']
  embeddings = model.encode(sentences)
  print(embeddings.shape)
  
  
if __name__ == '__main__':
  main()