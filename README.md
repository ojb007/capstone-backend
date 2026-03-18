# Capstone Financial NLP

금융 자연어처리(NLP) 캡스톤 프로젝트입니다.

## 프로젝트 구조

```
capstone_financial_nlp/
├─ app/                    # 프로젝트 코드
│  └─ api/                 # 데이터 처리, 학습, 평가 등 스크립트
├─ data/                   # 데이터셋 및 처리 결과
│  ├─ metadata/            # 데이터 설명 및 메타데이터
│  │  ├─ dataset_samples.txt
│  │  └─ dataset_summary.csv
│  ├─ processed/           # 처리된 데이터
│  │  └─ financial_sentiment.csv
│  ├─ raw/                 # 원본 데이터
│  │  ├─ 10k/              # 10-K 연간 보고서 TXT
│  │  ├─ finqa/
│  │  ├─ fiqa/
│  │  │  └─ fiqa_raw.csv
│  │  └─ fpb/
│  └─ unified/             # 통합 정리된 데이터
├─ results/                # 실험 결과
│  ├─ figures/             # 시각화/그래프 결과
│  ├─ models/              # 학습된 모델
│  └─ reports/             # 분석 보고서
├─ README.md
├─ .gitignore
└─ LICENSE
```

## 시작하기

```bash
pip install -r requirements.txt
```
