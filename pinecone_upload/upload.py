import os
import pandas as pd
from tqdm import tqdm
import openai
from pinecone import Pinecone, ServerlessSpec

# APIキーの設定（環境変数から）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "eiken-vocab"

# OpenAIの初期化（旧バージョン対応）
openai.api_key = OPENAI_API_KEY

# Pineconeの初期化（pinecone-client v3以降対応）
pc = Pinecone(api_key=PINECONE_API_KEY)

# Indexがなければ作成
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # ← Pineconeダッシュボードで表示されている region に合わせてください
        )
    )

index = pc.Index(INDEX_NAME)

# CSVファイルの読み込み
df = pd.read_csv("vocab_eiken_MVP.csv", encoding="shift_jis")

# ベクトル化してPineconeへ登録
batch = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    content = f"{row['word']} : {row['meaning_ja']} : {row['example']}"
    embedding = openai.Embedding.create(
        input=content,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    metadata = {
        "word": row["word"],
        "grade": row["grade"],
        "example": row["example"]
    }
    batch.append((str(row["id"]), embedding, metadata))

index.upsert(vectors=batch)
print("✅ Upload complete.")
