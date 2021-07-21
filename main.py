import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

import aiofiles

# from fastapi.staticfiles import StaticFiles
from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import nlp_wrapper
from lib import detaoperator
from lib.detaoperator import DetaBaseItem, read_preprocessed_txt_or_deta
from lib.detaoperator.detacon import DetaController
from lib.nlp import Similarity, Tfidf

# from lib.preprocessor import Wakatu

DATA: Final[Path] = Path("data/")
TXT_SAVED: Final[Path] = Path("data/tmp/")


app = FastAPI()


# app.mount('/static', StaticFiles(directory='static'), name='static')


templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root():
    return {"message": "Hello World"}


# NOTE RQ aiofiles.open
@lru_cache()
def read_staffs() -> List[str]:
    with open("./data/staff.txt") as f:  # NOTE aiofiles
        staffs = [s.rstrip() for s in f.readlines()]
    return staffs


@app.get("/l1", response_class=HTMLResponse)
async def l1(
    request: Request, staffs: List[str] = Depends(read_staffs), staff: Optional[str] = None
):
    answer = ""
    notification_style = ""

    if staff in staffs:
        answer = "います"
        notification_style = "is-success"
    elif staff:
        answer = "みつかりません"
        notification_style = "is-danger"

    context = {
        "request": request,
        "title": "Lec01",
        "subtitle": "staff.txt内を検索する",
        "notification_style": notification_style,
        "answer": answer,
    }
    return templates.TemplateResponse("./lec/l1.html.j2", context=context)


@app.get("/l2", response_class=HTMLResponse)
async def l2(request: Request):
    context = {
        "request": request,
        "title": "Lec02",
        "subtitle": "アップロードされたファイルどうしのTF-IDF値を算出する",
    }
    return templates.TemplateResponse("./lec/l2.html.j2", context=context)


@app.post("/l2", response_class=HTMLResponse)
async def post_l2(request: Request, file: UploadFile = File(...)):
    print("START:", start := time.time(), sep="\t")
    bstr = b""
    async for b in read_preprocessed_txt_or_deta(filename=file.filename, file=file):
        bstr += b
    preprocessed = bstr.decode("utf-8")

    index = [file.filename, *DEFAULT_DOCS]
    corpus = [preprocessed, *await default_docs_corpus()]
    df = Tfidf.tfidf_dataframe(index=index, corpus=corpus)
    table_style = ["table", "is-striped", "is-bordered"]
    table = df.to_html(border=0, classes=table_style)

    # change the color of row that is uploaded by user
    esc_fname = file.filename.replace(".", r"\.")
    p = fr"<tr>\s+(?=<th>{esc_fname})"  # `(?=...)` is lookahead assertion.
    tr_style = '<tr class="is-selected">'
    table = re.sub(p, tr_style, table, count=1)

    context = {
        "request": request,
        "title": "Lec02",
        "subtitle": "アップロードされたファイルどうしのTF-IDF値を算出する",
        "tfidf_table": table,
    }
    print("END:", end := time.time(), sep="\t")
    print("ELAPSED:", end - start, sep="\t")
    return templates.TemplateResponse("./lec/l2.html.j2", context=context)


DEFAULT_DOCS = ["hashire_merosu.txt", "kokoro.txt", "sanshiro.txt", "wagahaiwa_nekodearu.txt"]

# NOTE RQ
async def default_docs_corpus():
    items: List[DetaBaseItem] = []
    for fname in DEFAULT_DOCS:
        q = {"original_fname": fname}

        res = DetaController.base_fetch(query=q)
        resi: List[Dict[str, Any]] = res.items
        while res.last:  # turn over the pages
            resi += DetaController.base.fetch(last=res.last).items

        items += [DetaBaseItem(**i) for i in resi]

    corpus: List[str] = []
    for i in items:
        b = b""
        async for c in detaoperator.read_from_drive(i.path_on_drive, chunk_size=4096):
            b += c
        corpus.append(b.decode("utf-8"))
    return corpus


@app.get("/l3", response_class=HTMLResponse)
async def l3(request: Request, qword: Optional[str] = None):
    notification_style = None
    notification_msg = ""
    table = None

    if qword:
        corpus = await default_docs_corpus()
        simil_df = nlp_wrapper.word_docs_similarity(
            qword=qword, docnames=DEFAULT_DOCS, corpus=corpus
        )
        if simil_df is not None:
            notification_style = "is-success"
            notification_msg = "類似ドキュメントが見つかりました"

            table = simil_df.to_html(
                border=0,
                classes=["table", "is-striped", "is-bordered", "is-fullwidth"],
            )
        else:
            notification_style = "is-danger"
            notification_msg = "類似ドキュメントはありませんでした."

    context = {
        "request": request,
        "title": "Lec03",
        "subtitle": "入力されたワードと事前に保存されたドキュメントそれぞれとの類似度を計量する",
        "qword": qword,
        "notification_style": notification_style,
        "notification_msg": notification_msg,
        "simil_table": table,
    }
    return templates.TemplateResponse("./lec/l3.html.j2", context=context)
