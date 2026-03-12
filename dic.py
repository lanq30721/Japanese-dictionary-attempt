from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jaconv
import pandas as pd
import streamlit as st
import ipadic
from fugashi import Tagger
from jamdict import Jamdict


@dataclass
class KanjiAnnotation:
    kanji: str
    matched_reading: str
    on_readings: List[str]
    kun_readings: List[str]


def _get_jamdict() -> Jamdict:
    return Jamdict()


@st.cache_resource
def _get_tagger() -> Tagger:
    mecab_args = getattr(ipadic, "MECAB_ARGS", f"-d {ipadic.DICDIR}")
    return Tagger(mecab_args)


def _is_kana(ch: str) -> bool:
    return ("\u3040" <= ch <= "\u309f") or ("\u30a0" <= ch <= "\u30ff")


def _is_kanji(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _normalize_reading(reading: str) -> str:
    return reading.replace(".", "").replace("-", "")


def _expand_long_vowel(kana: str) -> str:
    vowels = {
        "あ": "あ", "い": "い", "う": "う", "え": "え", "お": "お",
        "か": "あ", "き": "い", "く": "う", "け": "え", "こ": "お",
        "さ": "あ", "し": "い", "す": "う", "せ": "え", "そ": "お",
        "た": "あ", "ち": "い", "つ": "う", "て": "え", "と": "お",
        "な": "あ", "に": "い", "ぬ": "う", "ね": "え", "の": "お",
        "は": "あ", "ひ": "い", "ふ": "う", "へ": "え", "ほ": "お",
        "ま": "あ", "み": "い", "む": "う", "め": "え", "も": "お",
        "や": "あ", "ゆ": "う", "よ": "お",
        "ら": "あ", "り": "い", "る": "う", "れ": "え", "ろ": "お",
        "わ": "あ", "ゐ": "い", "ゑ": "え", "を": "お",
        "が": "あ", "ぎ": "い", "ぐ": "う", "げ": "え", "ご": "お",
        "ざ": "あ", "じ": "い", "ず": "う", "ぜ": "え", "ぞ": "お",
        "だ": "あ", "ぢ": "い", "づ": "う", "で": "え", "ど": "お",
        "ば": "あ", "び": "い", "ぶ": "う", "べ": "え", "ぼ": "お",
        "ぱ": "あ", "ぴ": "い", "ぷ": "う", "ぺ": "え", "ぽ": "お",
        "きゃ": "あ", "きゅ": "う", "きょ": "お",
        "ぎゃ": "あ", "ぎゅ": "う", "ぎょ": "お",
        "しゃ": "あ", "しゅ": "う", "しょ": "お",
        "じゃ": "あ", "じゅ": "う", "じょ": "お",
        "ちゃ": "あ", "ちゅ": "う", "ちょ": "お",
        "にゃ": "あ", "にゅ": "う", "にょ": "お",
        "ひゃ": "あ", "ひゅ": "う", "ひょ": "お",
        "みゃ": "あ", "みゅ": "う", "みょ": "お",
        "りゃ": "あ", "りゅ": "う", "りょ": "お",
        "びゃ": "あ", "びゅ": "う", "びょ": "お",
        "ぴゃ": "あ", "ぴゅ": "う", "ぴょ": "お",
    }
    out: List[str] = []
    i = 0
    while i < len(kana):
        ch = kana[i]
        if ch == "ー":
            prev = out[-1] if out else ""
            out.append(vowels.get(prev, ""))
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def get_word_reading_hira(word: str) -> str:
    tagger = _get_tagger()
    tokens = list(tagger(word))
    reading = "".join(token.feature.kana or token.surface for token in tokens)
    return jaconv.kata2hira(reading)


def get_kanji_readings(kanji: str) -> Tuple[List[str], List[str]]:
    jam = _get_jamdict()
    res = jam.lookup(kanji)

    # 如果没查到这个汉字，直接返回空
    if not res.chars:
        return [], []

    char_data = res.chars[0]
    on_list = []
    kun_list = []

    # 官方正统的解析方式：从 rm_groups 里把音读和训读揪出来
    for rm_group in char_data.rm_groups:
        for reading in rm_group.readings:
            if reading.r_type == 'ja_on':  # ✅ 加上 r_
                on_list.append(reading.value)
            elif reading.r_type == 'ja_kun':  # ✅ 加上 r_
                kun_list.append(reading.value)

    # 用 jaconv 统一转换成平假名，并清理特殊符号
    on = [jaconv.kata2hira(_normalize_reading(r)) for r in on_list]
    kun = [jaconv.kata2hira(_normalize_reading(r)) for r in kun_list]

    return sorted(set(on)), sorted(set(kun))

def _match_candidate(reading: str, idx: int, candidate: str) -> Optional[str]:
    if not candidate:
        return None

    candidate = _expand_long_vowel(jaconv.kata2hira(candidate))
    prefix = ""
    if idx < len(reading) and reading[idx] == "っ":
        prefix = "っ"
        idx += 1

    if reading[idx: idx + len(candidate)] == candidate:
        return prefix + candidate
    return None


def annotate_word_kanji(word: str, reading: str) -> List[KanjiAnnotation]:
    idx = 0
    annotations: List[KanjiAnnotation] = []

    for ch in word:
        if _is_kana(ch):
            hira = jaconv.kata2hira(ch)
            if reading[idx: idx + len(hira)] == hira:
                idx += len(hira)
            continue

        if not _is_kanji(ch):
            continue

        on_readings, kun_readings = get_kanji_readings(ch)
        candidates = sorted(set(on_readings + kun_readings), key=len, reverse=True)

        matched = ""
        for candidate in candidates:
            picked = _match_candidate(reading, idx, candidate)
            if picked:
                matched = picked
                break

        if not matched and idx < len(reading):
            matched = reading[idx: idx + 1]

        if matched:
            idx += len(matched)

        annotations.append(
            KanjiAnnotation(
                kanji=ch,
                matched_reading=matched,
                on_readings=on_readings,
                kun_readings=kun_readings,
            )
        )

    return annotations


def build_rows(word: str) -> List[dict]:
    reading = get_word_reading_hira(word)
    annotations = annotate_word_kanji(word, reading)

    rows: List[dict] = []
    for item in annotations:
        rows.append(
            {
                "原汉字": item.kanji,
                "匹配读音": item.matched_reading or "-",
                "音读列表": " / ".join(item.on_readings) if item.on_readings else "-",
                "训读列表": " / ".join(item.kun_readings) if item.kun_readings else "-",
            }
        )
    return rows


def _map_pos_labels(pos_list: List[str]) -> List[str]:
    labels = set()
    for pos in pos_list:
        p = pos.lower()
        if "transitive verb" in p:
            labels.add("他动词")
        if "intransitive verb" in p:
            labels.add("自动词")
        if "noun" in p:
            labels.add("名词")
        if "adjective (keiyoushi)" in p:
            labels.add("形容词")
        if "adjectival nouns" in p or "keiyodoshi" in p:
            labels.add("形容动词")
    return sorted(labels)


def _collect_sense_details(entry) -> Tuple[List[str], List[str], List[str]]:
    pos_labels: List[str] = []
    gloss_cn_en: List[str] = []
    examples: List[str] = []
    for sense in entry.senses:
        labels = _map_pos_labels(sense.pos)
        pos_labels.extend(labels)

        gloss_items = []
        for g in sense.gloss:
            lang = getattr(g, "lang", "eng")
            text = getattr(g, "text", str(g))
            gloss_items.append(f"{lang}:{text}" if lang else text)
        if gloss_items:
            gloss_cn_en.append(" / ".join(gloss_items))

        if hasattr(sense, "examples") and sense.examples:
            for ex in sense.examples:
                if hasattr(ex, "jp"):
                    jp = getattr(ex, "jp", "")
                    en = getattr(ex, "en", "")
                    if jp or en:
                        examples.append(f"{jp} {en}".strip())
                else:
                    examples.append(str(ex))

    pos_labels = sorted(set(pos_labels))
    gloss_cn_en = list(dict.fromkeys(gloss_cn_en))
    examples = list(dict.fromkeys(examples))
    return pos_labels, gloss_cn_en, examples


def _collect_tri_language_gloss(entry) -> Tuple[str, str, str]:
    chinese: List[str] = []
    english: List[str] = []

    for sense in entry.senses:
        for g in sense.gloss:
            lang = getattr(g, "lang", "eng")
            text = getattr(g, "text", str(g))
            if not text:
                continue
            if lang in {"chi", "cmn", "zho", "zh"}:
                chinese.append(text)
            elif lang in {"eng", "en"}:
                english.append(text)

    kanji = " / ".join(k.text for k in entry.kanji_forms) if entry.kanji_forms else ""
    kana = " / ".join(k.text for k in entry.kana_forms) if entry.kana_forms else ""
    japanese = kanji or kana or "-"

    chinese_text = " / ".join(list(dict.fromkeys(chinese))) if chinese else "-"
    english_text = " / ".join(list(dict.fromkeys(english))) if english else "-"
    return chinese_text, english_text, japanese


def main() -> None:
    st.set_page_config(page_title="日语词典分析器", layout="wide")
    st.title("日语词典分析器")
    st.caption("输入日语单词，展示汉字匹配读音与音训读列表。")

    word = st.text_input("输入日语单词", value="学生", placeholder="例如: 学生、情報、時間")
    if not word.strip():
        st.info("请输入一个日语单词。")
        return

    normalized = word.strip()
    rows = build_rows(normalized)
    jam = _get_jamdict()
    res = jam.lookup(normalized)

    st.subheader("解析结果")
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("未识别到可分析的汉字。")

    st.subheader("释义")
    if res.entries:
        for entry_idx, entry in enumerate(res.entries, start=1):
            kanji = " / ".join(k.text for k in entry.kanji_forms) if entry.kanji_forms else "-"
            kana = " / ".join(k.text for k in entry.kana_forms) if entry.kana_forms else "-"
            pos_labels, gloss_cn_en, examples = _collect_sense_details(entry)

            with st.container(border=True):
                st.markdown(f"**{entry_idx}. {kanji}**  ({kana})")
                st.write("词性：" + (" / ".join(pos_labels) if pos_labels else "-") )
                if gloss_cn_en:
                    st.write("释义：" + " | ".join(gloss_cn_en))
                else:
                    st.write("释义：-")
                if examples:
                    st.write("例句：" + " / ".join(examples))
                else:
                    st.write("例句：-")
    else:
        st.info("未找到释义。")

    st.divider()
    st.subheader("三语对照")
    if res.entries:
        zh_text, en_text, ja_text = _collect_tri_language_gloss(res.entries[0])
    else:
        zh_text, en_text, ja_text = "-", "-", normalized

    col_zh, col_en, col_ja = st.columns(3)
    col_zh.markdown("**中文**")
    col_zh.write(zh_text)
    col_en.markdown("**英文**")
    col_en.write(en_text)
    col_ja.markdown("**日文**")
    col_ja.write(ja_text)


if __name__ == "__main__":
    main()