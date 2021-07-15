from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Similarity:
    @staticmethod
    def cos_simil(v1, v2, /):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class Tfidf:
    @staticmethod
    def tfidf(corpus: List[str]) -> Tuple[List[str], List[Any]]:
        # Disable L2 normalization. And match single-letter words also.
        vectorizer = TfidfVectorizer(norm=None, token_pattern=r"(?u)\b\w+\b")
        tfidf = vectorizer.fit_transform(corpus)

        columns: List[str] = sorted(d := vectorizer.vocabulary_, key=d.get)
        return columns, tfidf.toarray()

    @staticmethod
    def tfidf_dataframe(index: List[str], corpus: List[str]) -> pd.DataFrame:
        columns, data = Tfidf.tfidf(corpus)
        return pd.DataFrame(data, columns=columns, index=index)
