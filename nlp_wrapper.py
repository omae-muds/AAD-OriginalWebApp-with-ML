from typing import List, Union

import pandas as pd

from lib.nlp import Similarity, Tfidf
from lib.preprocessor import Wakatu


def word_docs_similarity(
    qword: str, docnames: List[str], corpus: List[str]
) -> Union[pd.DataFrame, None]:
    columns, data = Tfidf.tfidf(corpus=corpus)

    qwakati_list = Wakatu.parse_only_nouns_verbs_adjectives(sentense=qword).split()
    qvec = [int(col in qwakati_list) for col in columns]
    if any(qvec):
        df = pd.DataFrame(
            data=[Similarity.cos_simil(qvec, dvec) for dvec in data],
            index=docnames,
            columns=["similarity"],
        )
        return df.sort_values(by=df.columns[0], ascending=False)
    return None
