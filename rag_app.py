import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from typing import List

# --- 環境変数の読み込み ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# --- 必須環境変数チェック ---
required_vars = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "PINECONE_INDEX": PINECONE_INDEX
}
missing = [k for k, v in required_vars.items() if not v]
if missing:
    raise EnvironmentError(f"以下の環境変数が設定されていません: {', '.join(missing)}")

# --- クライアント初期化 ---
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# --- ベクトル生成 ---
def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

# --- Pineconeから文脈取得 ---
def retrieve_context(query: str, namespace: str = "vocab", top_k: int = 3) -> str:
    xq = get_embedding(query)
    res = index.query(
        vector=xq,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    contexts = [match.metadata.get("text", "") for match in res.matches if match.metadata]
    return "\n".join(contexts)

# --- GPTで回答生成 ---
def generate_answer(query: str, context: str) -> str:
    prompt = f"以下の文脈に基づいて、ユーザーの質問に日本語で簡潔に答えてください。\n\n文脈:\n{context}\n\n質問: {query}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは英検対策アプリのアシスタントです。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# --- namespace生成関数（級×モード）---
def get_namespace(level: str, mode: str) -> str:
    level_map = {
        "5級": "5", "4級": "4", "3級": "3", "準2級": "pre2"
    }
    mode_map = {
        "語彙": "vocab", "長文": "passages", "リスニング": "listening"
    }
    return f"{mode_map[mode]}-{level_map[level]}"

# --- Streamlit UI ---
st.set_page_config(page_title="英検RAGアプリ", layout="centered")
st.title("📘 英検RAGアプリ MVP")
st.caption("🎯 語彙・長文・リスニングに関する質問に、文脈付きで回答します。")

# --- サイドバー：級とモードの選択（Day-5 + Day-6）---
level = st.sidebar.selectbox("🔽 英検の級を選択してください", ["5級", "4級", "3級", "準2級"])
mode = st.sidebar.radio("📚 モードを選択してください", ["語彙", "長文", "リスニング"])

# --- 検索対象の明示 ---
st.caption(f"🧭 現在の検索対象：{level} × {mode}")

# --- 質問入力欄 ---
query = st.text_input("❓ 質問を入力してください（日本語／英語）")

# --- 質問履歴の初期化 ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- メイン処理 ---
if query:
    with st.spinner("検索中..."):
        try:
            namespace = get_namespace(level, mode)
            context = retrieve_context(query, namespace=namespace)
            answer = generate_answer(query, context)

            # 回答表示
            st.markdown("### ✅ 回答")
            st.write(answer)

            # 履歴に保存
            st.session_state.history.append((query, answer))
        except Exception as e:
            st.error(f"エラーが発生しました：{e}")

# --- 質問履歴表示（Day-5）---
if st.session_state.history:
    with st.expander("🕓 質問履歴"):
        for i, (q, a) in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
