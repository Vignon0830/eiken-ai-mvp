#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import openai
from dotenv import load_dotenv

# Pinecone SDK の新版インポート
from pinecone import Pinecone, ServerlessSpec

#───────────────────────────────────
# 環境変数
#───────────────────────────────────
load_dotenv()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")     # ex. "us-east1-gcp"
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")   # ex. "eiken-vocab"

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX]):
    raise RuntimeError("❌ .env に必要な変数が揃っていません。")

#───────────────────────────────────
# OpenAI
#───────────────────────────────────
openai.api_key = OPENAI_API_KEY

#───────────────────────────────────
# Pinecone 初期化
#───────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
index = pc.Index(PINECONE_INDEX)

#───────────────────────────────────
# テキスト→埋め込み取得
#───────────────────────────────────
def get_embedding(text: str) -> list[float]:
    resp = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return resp["data"][0]["embedding"]

#───────────────────────────────────
# CSV を自動エンコーディング判別でアップサート
#───────────────────────────────────
def upsert_csv(path: str, namespace: str, batch_size: int = 100):
    encodings = ("utf_8_sig", "utf-8", "shift_jis")
    f = None
    for enc in encodings:
        try:
            f = open(path, encoding=enc)
            # ヘッダー行を１行だけ読んでエラーが出ないかチェック
            _ = f.readline()
            f.seek(0)
            print(f"📑 `{path}` をエンコーディング `{enc}` でオープンしました")
            break
        except (UnicodeDecodeError, LookupError):
            if f:
                f.close()
            continue

    if f is None:
        raise RuntimeError(f"❌ `{path}` をいずれのエンコーディングでも開けませんでした")

    reader = csv.DictReader(f)
    batch  = []
    total  = 0

    for i, row in enumerate(reader):
        total += 1
        # カラム名に合わせて本文を取得
        text = row.get("text") or row.get("Text") or row.get("sentence") or ""
        if not text.strip():
            continue
        emb = get_embedding(text)
        batch.append((str(i), emb, row))
        if len(batch) >= batch_size:
            index.upsert(vectors=batch, namespace=namespace)
            batch.clear()

    if batch:
        index.upsert(vectors=batch, namespace=namespace)

    f.close()
    print(f"✅ `{path}` → namespace=`{namespace}` に {total} レコード upsert 完了")

#───────────────────────────────────
# メイン
#───────────────────────────────────
if __name__ == "__main__":
    targets = [
        ("vocab_eiken_MVP.csv",    "vocab"),
        ("passages_eiken_MVP.csv", "passages"),
        ("listen_eiken_MVP.csv",   "listening"),
    ]
    for csv_file, ns in targets:
        if not os.path.isfile(csv_file):
            print(f"⚠ `{csv_file}` が見つかりません。スキップします。")
            continue
        upsert_csv(csv_file, ns)
