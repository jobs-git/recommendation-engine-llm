# MIT License

# Copyright 2025 James Guana

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This code has been inspired by: https://www.kaggle.com/code/yashgoyal1605/recomendation-system

import pandas as pd
import copy
from uuid import uuid4
from utils import text_clean
from langchain.vectorstores import FAISS

class RECOMMEND ():

    def fit (self):
        self.train ()

    def train (self):
        uuids = [str(uuid4 ()) for _ in range(len(self.data["combined_text"].tolist()))]
        self.vectorstore = FAISS.from_texts(
            self.data["combined_text"].tolist(),  
            embedding=self.embeddings,
            metadatas=copy.deepcopy(self.metadatas),
            ids=uuids
        )

    def score (self, df):
        df = copy.deepcopy (df)
        return self.__recommendation_score (df)

    def predict (self, df, id):
        target_text_full = df.loc[df["pid"] == id, "combined_text"].values[0]
        target_text = text_clean(target_text_full)

        if not isinstance(target_text, str):
            raise ValueError("The target text must be a string.")

        seen_pids = set([id])

        def exclude_duplicates_filter (metadata):
            return metadata["pid"] not in seen_pids

        similar_docs = self.vectorstore.similarity_search(target_text, k = self.k, fetch_k = self.k, filter = exclude_duplicates_filter)

        extracted_data = []

        for doc in similar_docs:
            metadata = getattr(doc, "metadata", {})

            seen_pids.add(metadata.get("pid"))

            extracted_data.append({
                "pid": metadata.get("pid", "Unknown"),
                "product_name": metadata.get("product_name", "Unknown"),
                "product_category_tree": metadata.get("product_category_tree", "Unknown"),
                "discounted_price": metadata.get("discounted_price", "Unknown"),
                "brand": metadata.get("brand", "Unknown"),
            })

        similar_products = pd.DataFrame (extracted_data)

        return similar_products
    
    def set_k (self, value):
        self.k = value
        return self.k

    def get_vectorstore (self):
        return self.vectorstore

    def set_vectorstore (self, vectorstore):
        self.vectorstore = vectorstore
        return self.vectorstore

    def __init__ (self, data = None, metadatas = None, embeddings = None):
        self.embeddings = embeddings
        self.data = data
        self.metadatas = metadatas
        self.k = 10
        self.vectorstore = None
        pass

    def __extract_category (self, text):
        cleaned_text = text.strip("[]").strip("'").strip('"')
        try:
            return cleaned_text.split('>>')[0].strip()
        except IndexError:
            return "Unknown"

    def __calculate_performance(self, df, row):
        category = self.__extract_category(row['product_category_tree'])
        listA_df = self.predict(df, row['pid'])

        listA_df['category'] = listA_df['product_category_tree'].apply(self.__extract_category)
        matching_count = listA_df[listA_df['category'] == category].shape[0]
        
        performance = matching_count / len(listA_df) if len(listA_df) > 0 else 0
        return performance

    def __recommendation_score (self, df):
        df['performance'] = df.apply(lambda row: self.__calculate_performance (df, row), axis=1)
        average_performance = df['performance'].mean()
        return average_performance