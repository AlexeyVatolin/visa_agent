# Embedding Model Comparison: Mistral-embed vs Octen-Embedding-0.6B

**Date:** 2025-05-19
**Setup:** LangChain ChromaDB, MMR search (k=6, fetch_k=20)
**Corpus:** ~14K Telegram chat chunks about German visas from Serbia (Russian language)

## Questions

1. Как записаться на приём в TLS Contact для подачи на немецкий шенген из Сербии?
2. Какие документы нужны для подачи на немецкую шенгенскую визу из Сербии?
3. Можно ли податься на немецкую визу в Сербии без сербского ВНЖ (боравка)?
4. Каков процесс оформления визы по воссоединению семьи для Германии из Сербии?
5. Сколько времени занимает рассмотрение национальной визы в посольстве Германии в Белграде?
6. Как получить Blue Card (Blaue Karte) для Германии и какие требования?
7. По каким причинам чаще всего отказывают в немецкой визе при подаче из Сербии?
8. Нужен ли апостиль на диплом при подаче на рабочую визу в Германии в Белграде?
9. Можно ли получить визу на языковые курсы и потом сменить цель пребывания на поступление в вуз?
10. Какие требования по блокированному счёту (Sperrkonto) для национальной визы в Германию?

## Scores (0–1)

| Q# | Mistral-embed | Octen-Embedding-0.6B | Winner |
|----|:---:|:---:|:---:|
| 1  | 0.90 | 0.70 | Mistral |
| 2  | 0.85 | 0.85 | Tie |
| 3  | 0.75 | 0.90 | Octen |
| 4  | 0.65 | 0.75 | Octen |
| 5  | 0.45 | 0.75 | Octen |
| 6  | 0.75 | 0.85 | Octen |
| 7  | 0.55 | 0.55 | Tie |
| 8  | 0.70 | 0.90 | Octen |
| 9  | 0.60 | 0.70 | Octen |
| 10 | 0.50 | 0.65 | Octen |

## Summary

| Metric | Mistral-embed | Octen-Embedding-0.6B |
|--------|:---:|:---:|
| Total | 6.70 | **7.60** |
| Average | 0.67 | **0.76** |
| Wins | 1 | 7 |
| Ties | 2 | 2 |

## Decision

**Octen-Embedding-0.6B selected** — higher average relevance (+13%), wins 7/10 questions, runs locally via CPU (no API dependency or rate limits).
