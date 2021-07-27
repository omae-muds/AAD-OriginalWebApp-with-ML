import io
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Union

import aiofiles

# from fastapi.staticfiles import StaticFiles
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

import nlp_wrapper
from lib import detaoperator, imgps
from lib.detaoperator import DetaBaseItem, detacon, read_preprocessed_or_from_deta
from lib.detaoperator.detacon import DetaController
from lib.nlp import Tfidf

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
    b = b""
    async for b in read_preprocessed_or_from_deta(file=file, file_type="text"):
        b += b
    preprocessed = b.decode("shift-jis")

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

        res = detaoperator.query_base(query=q)
        if res:
            items += res

        # res = DetaController.base_fetch(query=q)
        # resi: List[Dict[str, Any]] = res.items
        # while res.last:  # turn over the pages
        #     resi += DetaController.base.fetch(last=res.last).items

        # items += [DetaBaseItem(**i) for i in resi]

    corpus: List[str] = []
    for i in items:
        b = b""
        async for c in detaoperator.read_from_drive(i.preprocessed_path, chunk_size=4096):
            b += c
        corpus.append(b.decode("shift-jis"))
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


@app.get("/l4", response_class=HTMLResponse)
async def l4(request: Request):
    return templates.TemplateResponse(
        "lec/l4.html.j2",
        context={
            "request": request,
            "title": "Lec04",
            "subtitle": "画像をアップロードする",
        },
    )


@app.post("/l4", response_class=HTMLResponse)
async def l4_post(request: Request, file: UploadFile = File(...)):
    imagename = file.filename
    # bimg = b""
    async for chunk in detaoperator.read_preprocessed_or_from_deta(file=file, file_type="image"):
        # bimg += chunk
        _ = ""

    item = detaoperator.query_one_base({"original_fname": imagename})
    if item is None:
        raise HTTPException(status_code=404)
    if d := item.other_data:
        features = d["akaze_features"]

    return templates.TemplateResponse(
        "lec/l4.html.j2",
        context={
            "request": request,
            "title": "Lec04",
            "subtitle": "画像をアップロードする",
            "features": features,
            "imagename": imagename,
        },
    )


@app.get("/srv", response_class=StreamingResponse)
async def srv(q: str, its: Optional[str] = None, useKey: Optional[bool] = None):
    # search key by fname
    if useKey:
        item = detaoperator.get_from_base(q)
    else:
        query = {"original_fname": q}
        item = detaoperator.query_one_base(query)

    if item is None or item.other_files is None:
        raise HTTPException(status_code=404, detail="Item not found")

    if its == "rawimg":
        path = item.other_files["raw"]
        mtype = "image/*"
    elif its == "akazeimg":
        path = item.other_files["akaze"]
        mtype = "image/*"
    elif its == "img":
        path = item.preprocessed_path
        mtype = "image/*"
    elif its in ["txt", "rawtxt"]:
        path = its + "/" + item.preprocessed_path
        mtype = "text/plain"

    b = b""
    async for chunk in detaoperator.read_from_drive(path):
        b += chunk

    if its == "txt":
        return StreamingResponse(b.decode("shift-jis"), media_type=mtype)

    return StreamingResponse(io.BytesIO(b), media_type=mtype)


@app.get("/storage", response_class=HTMLResponse)
async def redirect_fastapi(request: Request):
    dirs = ["akazeimg/", "img/", "rawimg/", "rawtxt/", "txt/"]
    return templates.TemplateResponse(
        "storage.html.j2",
        context={"request": request, "dirs": dirs},
    )


@app.get("/storage/{prefix}", response_class=HTMLResponse)
async def storage(
    request: Request,
    prefix: str,
    view: Optional[str] = None,
    targets: Optional[List[str]] = Query(None),
    act: Optional[str] = None,
):
    msg: Union[str, None] = None

    viewtxt: Optional[str] = None
    if view:
        if prefix in ["txt", "rawtxt"]:
            msg = f"選択されたファイルがBase上に見つかりませんでした: key == {view}"
            if item := detaoperator.get_from_base(view):
                path: Optional[str] = None
                if prefix == "txt":
                    path = item.preprocessed_path
                elif prefix == "rawtxt":
                    if d := item.other_files:
                        path = d["raw"]
                if path:
                    b = b""
                    async for chunk in detaoperator.read_from_drive(path):
                        b += chunk

                    viewtxt = b.decode("shift-jis")

                    msg = None

    if targets:
        if act == "comp":
            ...
        elif act == "del":
            failfiles = detaoperator.remove_files(targets)
            if failfiles:
                msg = f"次のファイルは削除に失敗しました:<br>{failfiles}"

    names = detaoperator.ls_drive(prefix)
    files: Optional[List[Dict[str, str]]] = None
    if names:
        if prefix == "akazeimg":
            delnum = len("akaze")
        elif prefix == "rawimg" or prefix == "rawtxt":
            delnum = len("raw")
        else:
            delnum = 0
        items = [detaoperator.query_one_base({"preprocessed_path": n[delnum:]}) for n in names]
        files = [
            {
                "name": name[len(prefix) + 1 :],
                "description": item.original_fname if item is not None else "",
            }
            for name, item in zip(names, items)
        ]

    return templates.TemplateResponse(
        "storage.html.j2",
        context={
            "request": request,
            "prefix": prefix,
            "view": view,
            "viewtxt": viewtxt,
            "msg": msg,
            "files": files,
        },
    )
