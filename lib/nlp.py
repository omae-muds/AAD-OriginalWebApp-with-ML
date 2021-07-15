from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Similarity:
    @staticmethod
    def cos_simil(v1, v2, /):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# TODO&REVIEW This is not used now. It's forever?, then remove.
# REVIEW about `source`
class Vecotorizer:
    @staticmethod
    def wakativector(
        q_wakatitas: List[str],
        source: Union[str, Path],
        calculator: Callable = Similarity.cos_simil,
    ):
        # source
        df = pd.read_csv(source, header=0, index_col=0)

        qvec = [int(col in q_wakatitas) for col in df.columns]

        if sum(qvec) == 0:
            return None
        ################
        simils = pd.DataFrame.from_dict(
            {name: calculator(qvec, df.loc[name].to_numpy()) for name in df.index},
            orient="index",
            columns=["similarity"],
        )
        simils = simils.sort_values(by=simils.columns[0], ascending=False)
        return simils.to_html(
            border=0,
            classes=["table", "is-striped", "is-bordered", "is-justify-content-center"],
        )


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
