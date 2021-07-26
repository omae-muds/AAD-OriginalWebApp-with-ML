import re
import urllib.request
from functools import lru_cache
from typing import List

import cv2
import MeCab
import numpy as np


class Aozora:
    """
     Get str from text in DL format from Aozora Bunko,
    cleansed to just the title, author, headings, and body.
    """

    # Pattern that matches Newline-code CR
    _p_carriagereturn = re.compile(r"\r")

    # matches the comment area between the two lines of `----` at the top of document
    # `re.MULTILINE` changes `^` and `$`
    # to match at the start and end of each line, not the whole string.
    _p_topcomment = re.compile(r"----+(.+\n+)+?----+", flags=re.MULTILINE)

    # matches the bibliographic information after 3 blank lines at the bottom of document.
    # !! It is faster to search from behind to match only the last one of the string.
    #   But no such option, so reverse the pattern, target, and return value.
    _p_btmbiblioinfo = re.compile(r"^\n(.+\n)+?(^\n){3}", flags=re.MULTILINE)  # already reversed

    # matches meta syntax such as headings and ruby(*furigana*)
    # It will be multiple. And it may be across two lines. So no `count` and `flags`.
    _p_meta = re.compile(r"(［[^［］]*?］)|(《[^《》]*?》)|\｜")

    @classmethod
    def cleansing(cls, raw: str) -> str:
        raw = cls._p_carriagereturn.sub("", raw)  # CRLF -> LF
        raw = cls._p_topcomment.sub("", raw, count=1)
        # Reverse pattern and return value to match only the last one more quickly
        raw = cls._p_btmbiblioinfo.sub("", raw[::-1], count=1)[::-1]
        return cls._p_meta.sub("", raw)


class Wakatu:
    """MeCab-Python3 wrapper"""

    # SlothLib's list of Japanese stop words.
    STOPWORDS_URL = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"

    _mtagger = MeCab.Tagger()

    @classmethod
    def parse_only_nouns_verbs_adjectives(cls, sentence: str) -> str:
        """Get Wakatigaki for specific word classes from Japanese sentence.

         This returns a parsed string that contain only nouns, verbs, and adjectives.
        But numerals, non-independent verbs, and suffixes are also removed.

        !! These are following Unidic. When using a different dictionary,
        such as ipadic, some changes are needed.
        """
        # Unidic style
        wclass_allowlist = ["名詞", "動詞", "形容詞"]
        wsubclass_denylist = ["数詞", "非自立可能", "接尾"]

        stopwords = cls._dl_stopwords()

        term: str
        parsed = ""
        node = cls._mtagger.parseToNode(sentence)
        while node:
            # These indices are following Unidic
            # HINT ipadic format:
            #     [wordClass, wSubClass1, wSubClass2, wSubClass3,
            #      conjugationType, cSubType, originalForm, reading, pronunciation]

            features: List[str] = node.feature.split(",")
            if features[0] in wclass_allowlist:
                if features[1] not in wsubclass_denylist:
                    # Get original form
                    if features[10] != "*":
                        term = features[10].strip()
                    else:
                        term = node.surface.strip()

                    if term not in stopwords:
                        parsed += " " + term

            node = node.next

        return parsed[1:]  # Exclude leading whitespace

    @classmethod
    @lru_cache(maxsize=None)
    def _dl_stopwords(cls) -> List[str]:
        """Download a list of Japanese stop words to a class variable."""
        with urllib.request.urlopen(cls.STOPWORDS_URL) as f:
            raw: str = f.read().decode("utf-8")
        return raw.splitlines()


class ImgNormalizer:
    # When split resize and equalize func, program should de/encode each time.
    @staticmethod
    def resize_and_equalize_hist(bimg: bytes, w: int = 256, h: int = 256) -> bytes:
        img = cv2.imdecode(np.frombuffer(bimg, np.uint8), cv2.IMREAD_COLOR)

        img = cv2.resize(img, (w, h))

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()
