# MIT License

# Copyright 2025 James Guana

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')

def cast_str(data):
    return str(data)

def text_clean(data):
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))

    lower = cast_str(data).lower()
    sub = re.sub(r'^[0-9]+', '', lower)
    tokens = tokenizer.tokenize(sub)
    dlist = [w for w in tokens if w.lower() not in stopword_set]

    clean_text = " ".join(dlist)
    return clean_text

def extract_category(text, level=0):
    cleaned_text = text.strip("[]").strip("'").strip('"')
    try:
        return cleaned_text.split('>>')[level].strip()
    except IndexError:
        return None