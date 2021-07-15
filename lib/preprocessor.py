import re
import urllib.request
from functools import lru_cache
from typing import Final, List

import MeCab


class Aozora:
    """
     Get str from text in DL format from Aozora Bunko,
    cleansed to just the title, author, headings, and body.
    """

    # # TODO Change to classmethod
    # def __init__(self, raw: str):
    #     # TODO To class variable likea `p_delcomment = re.comp...`
    #     # 先頭の2つの----行で囲まれたコメント領域の行を除く
    #     raw = re.sub(r"^----+(.+\n+)+?----+$", "", raw, count=1, flags=re.MULTILINE)
    #     # 最後の3行の空白行以降のコメント行を除く
    #     # 末尾から1つだけにマッチさせるため, rawとパターン, 返り値を逆順にしている
    #     raw = re.sub(r"^\n(.+\n)+?(^\n){3}", "", raw[::-1], count=1, flags=re.MULTILINE)[::-1]
    #     # 見出しやルビを示すメタ構文を削除
    #     p = re.compile(r"(［[^［］]*?］)|(《[^《》]*?》)|\｜")
    #     full_text = p.sub("", raw)

    #     self.full_text = full_text

    # Pattern that matches Newline code CR.
    _p_carriagereturn = re.compile(r"\r")

    #  Pattern that matches the comment area
    # between the two lines of `----` at the top of the document.
    #  `re.M` is same as `re.MULTILINE`; Changes `^` and `$` to
    # match at the start and end of each line, not the whole string.
    _p_topcomment = re.compile(r"----+(.+\n+)+?----+", flags=re.MULTILINE)

    # Pattern that matches the bibliographic information after 3 blank lines at the bottom of the document.
    # ! It is faster to search from behind to match only the last one of the string.
    #  But no such option, so reverse the pattern, target, and return value.
    _p_btmbiblioinfo = re.compile(r"^\n(.+\n)+?(^\n){3}", flags=re.MULTILINE)  # already reversed

    # Patterns that matches meta syntax such as headings and rubrics.
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
    """Get "Wakatigaki" from Japanese text."""

    # SlothLib's list of Japanese stop words.
    STOPWORDS_URL: Final[
        str
    ] = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"

    _stopwords: List[str] = []

    wclass_denylist = ["数", "非自立", "接尾"]
    wclass_allowlist = ["名詞", "動詞", "形容詞"]

    # REVIEW Realy need to be a class-variable?
    _mtagger = MeCab.Tagger()
    # REVIEW Realy need?
    _mtagger.parse("")  # To avoid UnicodeDecodeError?

    @classmethod
    def parse_only_nouns_verbs_adjectives(cls, sentense: str) -> str:
        """Get "Wakatigaki" from Japanese text.

         This returns a parsed string that contain only nouns, verbs, and adjectives.
        But numerals, non-independent verbs, and suffixes are also removed.
        """
        # FIXME This will be running by each calling
        # NOTE Metaclass, Temp, Auto load Standing Server
        if not cls._stopwords:
            cls._stopwords = cls._dl_stopwords()

        term: str
        wakatigaki = ""
        node = cls._mtagger.parseToNode(sentense)
        while node:
            # [品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音]
            # [wordClass, subclass1, subclass2, subclass3, conjugationType,
            #  conjugationSubType, originalForm, reading, pronunciation]
            features: List[str] = node.feature.split(",")

            # Get the original form of a word
            if features[6] != "*":
                term = features[6].strip()
            else:
                term = node.surface.strip()

            # Skip adding term to wakatigaki
            if term in cls._stopwords or features[1] in cls.wclass_denylist:
                node = node.next
                continue

            if features[0] in cls.wclass_allowlist:
                wakatigaki += " " + term

            node = node.next

        return wakatigaki[1:]  # Exclude leading whitespace

    @classmethod
    @lru_cache(maxsize=None)
    def _dl_stopwords(cls) -> List[str]:
        """Download a list of Japanese stop words to a class variable."""
        with urllib.request.urlopen(cls.STOPWORDS_URL) as f:
            raw: str = f.read().decode("utf-8")
        return raw.splitlines()
