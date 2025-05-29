# Streamlit 디시인사이드 텍스트 분석 대시보드
# ✅ 2025-05-29 기준 업그레이드 내역 요약
# - 최신 openai.chat.completions.create API 방식 적용
# - API 키를 코드 내 직접 입력 가능하도록 수정
# - 워드클라우드용 폰트를 Windows 환경에 맞게 수정
# - GPT 해석 오류 처리 완료 (message.content 접근)
# - 사용자가 직접 '등록어', '불용어' 추가 가능
# - 등록 후 전체 분석 결과 재실행 가능하도록 흐름 개선
# - 등록된 키워드는 txt 파일에 자동 저장됨
# - PDF 보고서 한글 인코딩 오류 해결: 한글 폰트 등록

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import openai
import os
from fpdf import FPDF

# GPT API Key 설정
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# 불용어 및 보존 키워드 파일 경로 (바탕화면)
stopwords_path = os.path.expanduser("~/Desktop/stopwords.txt")
preserve_path = os.path.expanduser("~/Desktop/preserve_keywords.txt")

def load_keywords():
    with open(stopwords_path, encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)
    with open(preserve_path, encoding="utf-8") as f:
        preserve_keywords = set(line.strip() for line in f)
    return stopwords, preserve_keywords

stopwords, preserve_keywords = load_keywords()

st.title("🧠 디시인사이드 텍스트 분석 대시보드")

uploaded_file = st.file_uploader("📂 CSV 파일 업로드 (title 컬럼 포함)", type=["csv"])

st.sidebar.title("🔧 키워드 관리")
add_stop = st.sidebar.text_input("불용어 추가")
add_preserve = st.sidebar.text_input("등록어 추가")
update_keywords = st.sidebar.button("📌 키워드 업데이트 및 분석 재실행")

if update_keywords:
    if add_stop:
        with open(stopwords_path, "a", encoding="utf-8") as f:
            f.write(add_stop.strip() + "\n")
    if add_preserve:
        with open(preserve_path, "a", encoding="utf-8") as f:
            f.write(add_preserve.strip() + "\n")
    st.sidebar.success("키워드가 저장되었으며 분석이 재실행됩니다. 다시 파일을 업로드 해주세요.")
    st.stop()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 데이터 업로드 완료")

    okt = Okt()

    def extract_nouns(text):
        nouns = okt.nouns(text)
        return [n for n in nouns if n in preserve_keywords or (n not in stopwords and len(n) > 1)]

    df['tokens'] = df['title'].astype(str).apply(extract_nouns)
    texts = df['tokens'].tolist()
    titles = df['title'].tolist()

    st.subheader("📊 간단한 분석 결과")
    st.markdown(f"- 전체 문서 수: **{len(df)}**")
    total_words = sum(len(t) for t in texts)
    unique_words = len(set(word for t in texts for word in t))
    st.markdown(f"- 추출된 전체 단어 수: **{total_words}**, 고유 단어 수: **{unique_words}**")
    sample_keywords = list(set(word for t in texts for word in t))[:10]
    st.markdown(f"- 예시 단어: {', '.join(sample_keywords)}")

    report_text = f"""총 문서 수: {len(df)}
전체 단어 수: {total_words}
고유 단어 수: {unique_words}
예시 단어: {', '.join(sample_keywords)}\n"""

    st.subheader("☁️ 워드클라우드")
    all_nouns = [noun for text in texts for noun in text]
    word_freq = Counter(all_nouns)
    report_text += f"""\n워드클라우드 주요 단어:\n{word_freq.most_common(10)}\n"""
    wcloud = WordCloud(font_path="C:/Windows/Fonts/malgun.ttf", background_color='white', width=800, height=400)
    wcloud.generate_from_frequencies(word_freq)
    fig_wc = plt.figure(figsize=(10, 5))
    plt.imshow(wcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig_wc)

    if st.button("🧠 GPT로 워드클라우드 해석"):
        prompt = f"다음 단어 빈도를 바탕으로 주요 이슈나 주제를 요약해 주세요: {word_freq.most_common(20)}"
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown(response.choices[0].message.content)

    st.subheader("🔗 N-gram 네트워크")
    bigrams = list(combinations(all_nouns, 2))
    bigram_freq = Counter(bigrams)
    G = nx.Graph()
    for (a, b), freq in bigram_freq.items():
        if freq >= 3 and a != b:
            G.add_edge(a, b, weight=freq)

    pos = nx.spring_layout(G, k=0.6, seed=42)
    fig_ng, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_family='Malgun Gothic', font_size=10)
    st.pyplot(fig_ng)

    st.subheader("🌐 중심성 분석")
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)

    centrality_df = pd.DataFrame({
        "단어": list(degree.keys()),
        "Degree": [round(degree[n], 4) for n in degree],
        "Betweenness": [round(betweenness[n], 4) for n in degree],
        "Closeness": [round(closeness[n], 4) for n in degree],
        "Eigenvector": [round(eigenvector[n], 4) for n in degree]
    }).sort_values(by="Degree", ascending=False)

    st.dataframe(centrality_df.head(20))
    report_text += f"""\n중심성 상위 단어:\n{centrality_df.head(10).to_string(index=False)}\n"""

    if st.button("🧠 GPT로 중심성 해석"):
        prompt = f"""다음은 단어 중심성 분석 결과입니다. 중요한 키워드와 의미를 요약해 주세요:\n{centrality_df.head(10).to_string(index=False)}"""
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown(response.choices[0].message.content)

    st.subheader("📚 LDA 토픽 모델링")
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10, random_state=42)

    topic_keywords = {}
    report_text += """\nLDA 토픽 키워드:\n"""
    for idx, topic in ldamodel.show_topics(num_words=5, formatted=False):
        keywords = ", ".join([word for word, prob in topic])
        st.markdown(f"**토픽 {idx + 1}:** {keywords}")
        topic_keywords[f"Topic {idx + 1}"] = [word for word, _ in topic]
        report_text += f"Topic {idx + 1}: {keywords}\n"

    if st.button("🧠 GPT로 토픽 해석"):
        prompt = """다음은 LDA 토픽 모델링 키워드입니다. 각 토픽의 주제를 요약하고 유추 가능한 이슈를 설명해 주세요:\n"""
        for k, v in topic_keywords.items():
            prompt += f"{k}: {', '.join(v)}\n"
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown(response.choices[0].message.content)

    pdf = FPDF()
    pdf.add_page()
    font_path = "C:/Windows/Fonts/malgun.ttf"
    pdf.add_font('Malgun', '', font_path, uni=True)
    pdf.set_font("Malgun", '', 12)
    for line in report_text.split("\n"):
        pdf.cell(0, 10, txt=line, ln=True)

    pdf_path = "analysis_report.pdf"
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button("📄 PDF 보고서 다운로드", f, file_name="analysis_report.pdf", mime="application/pdf")
