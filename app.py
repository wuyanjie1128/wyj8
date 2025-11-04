# app.py
# -*- coding: utf-8 -*-

import os
import io
import time
import math
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# 금융 데이터: KRX/KOSPI200
from pykrx import stock

# 모델/스코어링 유틸
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------------
# 기본 설정 & 안전 가드
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="코스피200 주식 추천 시스템",
    page_icon="📈",
    layout="wide",
)

# .env 로드 (있으면)
load_dotenv()

# 세션 상태 초기화
def _init_session():
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {
            "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        }
    if "last_run" not in st.session_state:
        st.session_state.last_run = None

_init_session()

# 공용 상수
KOSPI200_INDEX_CODE = "1028"  # KOSPI200의 KRX 지수 코드
DATE_FMT = "%Y%m%d"

# 안전한 날짜 변환
def to_yyyymmdd(d: dt.date) -> str:
    return d.strftime(DATE_FMT)

def last_business_day(d: dt.date) -> dt.date:
    # pykrx는 휴장일에 대한 자동 보정이 없으므로, 최근 10영업일까지 후퇴 탐색
    for i in range(0, 12):
        probe = d - dt.timedelta(days=i)
        try:
            tickers = stock.get_index_portfolio_deposit_file(KOSPI200_INDEX_CODE, to_yyyymmdd(probe))
            if isinstance(tickers, list) and len(tickers) > 0:
                return probe
        except Exception:
            pass
    # 그래도 실패하면 그냥 어제 날짜
    return d - dt.timedelta(days=1)

# 캐시: 지수 구성 종목
@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_kospi200_tickers(asof: dt.date) -> pd.DataFrame:
    date_str = to_yyyymmdd(asof)
    codes = stock.get_index_portfolio_deposit_file(KOSPI200_INDEX_CODE, date_str)
    names = [stock.get_market_ticker_name(c) for c in codes]
    df = pd.DataFrame({"티커": codes, "종목명": names})
    return df

# 캐시: 개별 OHLCV
@st.cache_data(show_spinner=False, ttl=60 * 30)
def get_price_df(ticker: str, start: str, end: str) -> pd.DataFrame:
    # pykrx: get_market_ohlcv_by_date(start, end, ticker)
    try:
        df = stock.get_market_ohlcv_by_date(start, end, ticker)
        df = df.rename(
            columns={
                "시가": "Open",
                "고가": "High",
                "저가": "Low",
                "종가": "Close",
                "거래량": "Volume",
                "거래대금": "Value",
            }
        )
        # 인덱스를 날짜 컬럼으로
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        # 빈 DF 반환 (다운스트림에서 안전 처리)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Value"])

# 캐시: 펀더멘털 (PER/PBR/ROE)
@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_fundamental(asof: str) -> pd.DataFrame:
    # 시장 전체를 받아오고 나중에 KOSPI200만 필터
    try:
        kospi = stock.get_market_fundamental_by_ticker(asof, market="KOSPI")
        kosdaq = stock.get_market_fundamental_by_ticker(asof, market="KOSDAQ")
        df = pd.concat([kospi, kosdaq], axis=0)
        df.index.name = "티커"
        df.reset_index(inplace=True)
        # 컬럼 표준화
        # pykrx 반환: BPS, PER, PBR, EPS, DIV, DPS, ROE 등
        keep = ["티커", "PER", "PBR", "ROE"]
        out = df[keep].copy()
        return out
    except Exception:
        return pd.DataFrame(columns=["티커", "PER", "PBR", "ROE"])

# 스코어 유틸
def zscore(series: pd.Series) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).astype(float)
    if s.std(ddof=0) == 0 or s.dropna().empty:
        return pd.Series(np.zeros(len(s)), index=s.index)
    scaler = StandardScaler(with_mean=True, with_std=True)
    arr = s.fillna(s.median()).to_numpy().reshape(-1, 1)
    z = scaler.fit_transform(arr).flatten()
    return pd.Series(z, index=s.index)

# ------------------------------------------------------------------------------------
# 사이드바: 설정 / API 인증 정보 / 분석 설정
# ------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    st.caption("필수/선택 항목을 설정한 뒤, 아래 **분석 실행** 버튼을 클릭하세요.")

    # 데이터 구간 설정
    today = dt.date.today()
    end_date = last_business_day(today)
    default_start = end_date - dt.timedelta(days=365)  # 기본 1년

    st.subheader("데이터 기간")
    colA, colB = st.columns(2)
    with colA:
        start_input = st.date_input("시작일", default_start, max_value=end_date)
    with colB:
        end_input = st.date_input("종료일", end_date, max_value=end_date)

    st.subheader("API 인증 정보 (선택)")
    st.caption("필수는 아닙니다. 다른 데이터/모델 확장을 위해 보관합니다. 환경변수(.env)도 지원합니다.")
    alpha_key = st.text_input(
        "Alpha Vantage API Key (선택)",
        value=st.session_state.api_keys.get("ALPHA_VANTAGE_API_KEY", ""),
        type="password",
        help="미입력해도 작동합니다. 추후 대체 데이터 소스 연동용.",
    )
    openai_key = st.text_input(
        "OpenAI API Key (선택)",
        value=st.session_state.api_keys.get("OPENAI_API_KEY", ""),
        type="password",
        help="추천 설명 자동생성 등 고급 기능 확장용 (현재 필수 아님).",
    )
    st.session_state.api_keys["ALPHA_VANTAGE_API_KEY"] = alpha_key
    st.session_state.api_keys["OPENAI_API_KEY"] = openai_key

    st.subheader("분석 설정")
    st.caption("팩터 가중치와 필터를 조정해 보세요.")
    n_pick = st.number_input("선정 종목 수", min_value=5, max_value=50, value=20, step=1)
    min_value_krw = st.number_input("일평균 거래대금(백만원) 최소", min_value=0, value=10, step=5,
                                    help="최근 20영업일 평균 기준. 유동성 필터.")
    st.markdown("---")
    st.markdown("**팩터 가중치 (합은 자동 정규화)**")
    w_mom = st.slider("모멘텀(6개월-1개월)", 0.0, 1.0, 0.40, 0.05)
    w_vol = st.slider("저변동성(20일)", 0.0, 1.0, 0.20, 0.05)
    w_val = st.slider("가치(PBR 역수)", 0.0, 1.0, 0.25, 0.05)
    w_qlt = st.slider("퀄리티(ROE)", 0.0, 1.0, 0.15, 0.05)

    # 실행 버튼
    run_button = st.button("🚀 분석 실행", use_container_width=True)

st.title("📈 코스피200 주식 추천 시스템")
st.write(
    "코스피200 구성 종목을 대상으로 **모멘텀·저변동성·가치·퀄리티** 팩터를 결합한 스코어를 계산하여 "
    f"상위 {n_pick}개 종목을 추천합니다. (한글 UI / 웹 버전)"
)

# ------------------------------------------------------------------------------------
# 본문: 분석 파이프라인
# ------------------------------------------------------------------------------------
def compute_liquidity(df_price: pd.DataFrame) -> float:
    """최근 20영업일 평균 거래대금(백만원 단위)"""
    if df_price.empty or "Value" not in df_price.columns:
        return 0.0
    tail = df_price["Value"].tail(20)
    if tail.empty:
        return 0.0
    # pykrx Value는 원화 단위로 제공
    return float(np.nanmean(tail) / 1_000_000.0)

def compute_momentum(df_price: pd.DataFrame) -> float:
    """
    모멘텀(6개월-1개월): 최근 1개월은 제외하고 그 이전 5개월(≈105영업일) 수익률.
    데이터가 부족하면 NaN.
    """
    if df_price.empty or "Close" not in df_price.columns:
        return np.nan
    closes = df_price["Close"].dropna()
    if len(closes) < 140:
        return np.nan
    ret_6m = closes.iloc[-22-110]  # 대략 6M 전 근처 값
    last_1m = closes.iloc[-22]     # 1M 전 근처 값
    now = closes.iloc[-1]
    try:
        # (6M→1M) 구간 수익률
        return (last_1m / ret_6m) - 1.0
    except Exception:
        return np.nan

def compute_volatility(df_price: pd.DataFrame) -> float:
    """최근 20일 일간수익률 표준편차(낮을수록 우수) → 부호 반전하여 점수화용"""
    if df_price.empty or "Close" not in df_price.columns:
        return np.nan
    closes = df_price["Close"].dropna()
    if len(closes) < 25:
        return np.nan
    ret = closes.pct_change().dropna().tail(20)
    if ret.empty:
        return np.nan
    return -float(ret.std())  # 낮은 변동성 선호 → 부호 반전

def safe_inverse(x: float) -> float:
    if x in [None, np.nan]:
        return np.nan
    try:
        if x == 0:
            return np.nan
        return 1.0 / float(x)
    except Exception:
        return np.nan

def normalized_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(w.values())
    if s <= 0:
        # 기본 분배
        return {k: 1.0 / len(w) for k in w}
    return {k: v / s for k, v in w.items()}

def build_scores(
    base_universe: pd.DataFrame,
    start_str: str,
    end_str: str,
    min_liquidity_million: float,
    weights: Dict[str, float],
) -> pd.DataFrame:

    results = []
    tickers = base_universe["티커"].tolist()
    names_map = dict(zip(base_universe["티커"], base_universe["종목명"]))

    progress = st.progress(0.0, text="데이터 수집 중...")
    steps = max(len(tickers), 1)

    # 가격 기반 팩터 계산
    price_cache: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers, start=1):
        dfp = get_price_df(t, start_str, end_str)
        price_cache[t] = dfp

        liq = compute_liquidity(dfp)
        mom = compute_momentum(dfp)
        vol = compute_volatility(dfp)

        results.append(
            {
                "티커": t,
                "종목명": names_map.get(t, t),
                "유동성(백만원,20일평균)": liq,
                "모멘텀(6-1M)": mom,
                "저변동성(20일)": vol,
            }
        )
        if i % 5 == 0 or i == steps:
            progress.progress(i / steps, text=f"데이터 수집 중... ({i}/{steps})")

    price_factors = pd.DataFrame(results)

    # 유동성 필터
    filtered = price_factors[price_factors["유동성(백만원,20일평균)"] >= min_liquidity_million].copy()
    if filtered.empty:
        st.warning("유동성 필터 이후 남는 종목이 없습니다. 조건을 완화해 주세요.")
        return pd.DataFrame()

    # 펀더멘털 결합
    st.info("펀더멘털(PER/PBR/ROE) 수집 중...")
    fnda = get_fundamental(end_str)
    merged = pd.merge(filtered, fnda, on="티커", how="left")

    # 각 팩터 점수화(표준화 Z)
    merged["모멘텀_Z"] = zscore(merged["모멘텀(6-1M)"])
    merged["저변동성_Z"] = zscore(merged["저변동성(20일)"])
    # 가치: PBR의 역수(낮은 PBR 선호)
    merged["가치(1/PBR)"] = merged["PBR"].apply(safe_inverse)
    merged["가치_Z"] = zscore(merged["가치(1/PBR)"])
    # 퀄리티: ROE 높을수록 선호
    merged["퀄리티_Z"] = zscore(merged["ROE"])

    # 가중치 정규화 후 총합 스코어
    w = normalized_weights(weights)
    merged["종합스코어"] = (
        merged["모멘텀_Z"] * w["momentum"]
        + merged["저변동성_Z"] * w["lowvol"]
        + merged["가치_Z"] * w["value"]
        + merged["퀄리티_Z"] * w["quality"]
    )

    # 최근 수익률 참고(보여주기용)
    def last_1m_ret(tk: str) -> float:
        dfp = price_cache.get(tk, pd.DataFrame())
        if dfp.empty or len(dfp) < 23:
            return np.nan
        c = dfp["Close"].dropna()
        try:
            return c.iloc[-1] / c.iloc[-22] - 1.0
        except Exception:
            return np.nan

    merged["최근1개월수익률"] = merged["티커"].apply(last_1m_ret)

    # 정렬
    merged.sort_values(["종합스코어", "최근1개월수익률"], ascending=[False, False], inplace=True)

    # 열 정리
    cols = [
        "티커", "종목명",
        "종합스코어",
        "모멘텀(6-1M)", "저변동성(20일)", "가치(1/PBR)", "ROE",
        "모멘텀_Z", "저변동성_Z", "가치_Z", "퀄리티_Z",
        "최근1개월수익률",
        "유동성(백만원,20일평균)",
        "PER", "PBR",
    ]
    for c in cols:
        if c not in merged.columns:
            merged[c] = np.nan

    return merged[cols].reset_index(drop=True)

# ------------------------------------------------------------------------------------
# 실행
# ------------------------------------------------------------------------------------
if run_button:
    try:
        st.session_state.last_run = dt.datetime.now()
        start_str = to_yyyymmdd(start_input)
        end_str = to_yyyymmdd(end_input)

        st.subheader("1) 유니버스: 코스피200 구성 종목")
        asof = last_business_day(end_input)
        uni = get_kospi200_tickers(asof)
        st.dataframe(uni, use_container_width=True, height=240)

        st.subheader("2) 팩터 스코어 계산")
        weights = {
            "momentum": float(w_mom),
            "lowvol": float(w_vol),
            "value": float(w_val),
            "quality": float(w_qlt),
        }

        result = build_scores(
            base_universe=uni,
            start_str=start_str,
            end_str=end_str,
            min_liquidity_million=float(min_value_krw),
            weights=weights,
        )

        if not result.empty:
            st.success("스코어 계산 완료!")
            st.markdown(f"**상위 {n_pick} 종목 추천** (종합스코어 기준)")
            topn = result.head(n_pick).copy()
            st.dataframe(topn, use_container_width=True, height=480)

            # 다운로드
            csv = topn.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ 추천 결과 CSV 다운로드",
                data=csv,
                file_name=f"kospi200_reco_{end_str}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # 간단 백테스트(참고용): 동차익일 매수 가정 X, 단순 현재까지 수익률 컬럼 요약
            st.subheader("3) 간단 요약 지표")
            st.write(
                "- 스코어는 표준화(Z) 기반 가중 합산입니다.\n"
                "- 저변동성은 20일 표준편차의 음수 값(낮을수록 좋음)으로 점수화했습니다.\n"
                "- 가치는 PBR의 역수를 사용했습니다(낮은 PBR 선호).\n"
                "- 유동성 필터: 최근 20영업일 평균 거래대금이 설정값 이상인 종목만 포함합니다."
            )

        else:
            st.error("결과가 비어 있습니다. 기간/유동성/가중치를 조정해 보세요.")

    except Exception as e:
        st.error("예상치 못한 오류가 발생했습니다. 설정을 조정하거나 기간을 변경해 보세요.")
        st.exception(e)
else:
    st.info("좌측 사이드바에서 기간/가중치/필터를 설정한 뒤 **분석 실행**을 눌러주세요.")

# 푸터
st.markdown("---")
st.caption(
    "※ 본 도구는 교육/리서치 목적입니다. 투자 판단의 최종 책임은 사용자에게 있으며, "
    "실거래 적용 전 반드시 추가 검증을 진행하세요."
)
