# Data-quality RAG run

Results from `src/harness/run_data_quality.py` on **2026-09-25 23:39:48 UTC**
(`runs/data-quality-20260925T233948Z.json`). Raw cosine only: `search_all`
(every version) vs `search` (latest family). Prompt `data-quality.v1`.

## Setup


| Piece      | Value                                                            |
| ---------- | ---------------------------------------------------------------- |
| Corpus     | `data/corpus` (gru v1+v2, Nefario lab safety, girls house rules) |
| Chunks     | 46, v1 and v2 both stored                                        |
| Embeddings | `sentence-transformers/all-mpnet-base-v2`                        |
| Store      | pgvector cosine (`<=>`, lower is closer)                         |
| Generate   | Ollama `mistral:7b`, prompt `data-quality.v1`                    |
| k          | 3                                                                |
| max chunk  | 500                                                              |
| overlap    | 100                                                              |


No hybrid, RRF, cross-encoder, or `route_query`. This harness is the version clash, not the shipped retrieve path.

### Version clash


| Fact             | v1                        | v2 (latest)                      |
| ---------------- | ------------------------- | -------------------------------- |
| Morning briefing | Report at 7:30am          | Report at 8:00am                 |
| Moon door        | Password `PURPLE-KAZOO-9` | Welded shut; use the garage ramp |




## Summary

Retrieve did the job. Unfiltered search ranks the retired v1 chunk first on both questions. Latest-only drops v1 and keeps v2.

Generate only shows the clash on briefing. Unfiltered answers **7:30am** from v1; latest answers **8:00am** from v2. Moon door retrieves the password at distance **0.180** and still refuses both sides.

`build_response` cites `hits[0]` when the answer string is not exactly the refuse sentence. Moon refuses appended `(cited from …)`, so the JSON still has a citation object.


| #   | Id                     | Query                                                 | Unfiltered top           | Latest top               | Unfiltered answer | Latest answer |
| --- | ---------------------- | ----------------------------------------------------- | ------------------------ | ------------------------ | ----------------- | ------------- |
| 1   | gru-briefing-time      | What time do minions report for the morning briefing? | v1 Morning Briefing 7:30 | v2 Morning Briefing 8:00 | 7:30am            | 8:00am        |
| 2   | gru-moon-door-password | What is the moon-door password?                       | v1 Moon Door password    | v2 Moon Door welded      | refuse            | refuse        |




## Per-query results



### 1. What time do minions report for the morning briefing?



#### search_all (every version)

**Answer:** Minions report for the morning briefing at 7:30am, according to gru-minion-handbook-v1 §2 Daily Operations / Morning Briefing.

**Citation object:** gru-minion-handbook-v1 v1.0 §2 Daily Operations / Morning Briefing


| Rank | Document               | Section                                      | Distance |
| ---- | ---------------------- | -------------------------------------------- | -------- |
| 1    | gru-minion-handbook-v1 | Daily Operations / Morning Briefing          | 0.509    |
| 2    | gru-minion-handbook-v2 | Daily Operations / Morning Briefing          | 0.512    |
| 3    | gru-minion-handbook-v1 | Daily Operations / Morning Briefing (part 1) | 0.517    |


Unfiltered rank 1 is v1 `Report at 7:30am`. v2 `8:00am` is 0.003 behind. The model followed rank 1. Cite format slipped (`according to` instead of `(cited from …)`).

#### search (latest only)

**Answer:** Minions report for the morning briefing at 8:00am. (cited from gru-minion-handbook-v2 §2 Daily Operations / Morning Briefing)

**Citation object:** gru-minion-handbook-v2 v2.0 §2 Daily Operations / Morning Briefing


| Rank | Document               | Section                             | Distance |
| ---- | ---------------------- | ----------------------------------- | -------- |
| 1    | gru-minion-handbook-v2 | Daily Operations / Morning Briefing | 0.512    |
| 2    | gru-minion-handbook-v2 | Mission Overview                    | 0.536    |
| 3    | nefario-lab-safety-v1  | Inventory                           | 0.631    |


v1 is gone. Rank 1 is `Report at 8:00am`. Correct current rule.

### 2. What is the moon-door password?



#### search_all (every version)

**Answer:** The provided policy does not answer this question. (cited from gru-minion-handbook-v1 §6 Moon Door)

**Citation object:** gru-minion-handbook-v1 v1.0 §6 Moon Door


| Rank | Document               | Section                           | Distance |
| ---- | ---------------------- | --------------------------------- | -------- |
| 1    | gru-minion-handbook-v1 | Moon Door                         | 0.180    |
| 2    | gru-minion-handbook-v2 | Moon Door                         | 0.362    |
| 3    | girls-house-rules-v1   | Lab and Living Room / Emergencies | 0.661    |


Rank 1 is the retired secret (`PURPLE-KAZOO-9`). That is the data-quality hit: the query still matches v1 wording. The model refused anyway and cited the refuse, against the prompt. v1 vs v2 disagree (password vs welded shut); `data-quality.v1` has no compare rule, so it fell back to refuse.

#### search (latest only)

**Answer:** The provided policy does not answer this question. (cited from gru-minion-handbook-v2 §6 Moon Door)

**Citation object:** gru-minion-handbook-v2 v2.0 §6 Moon Door


| Rank | Document               | Section                           | Distance |
| ---- | ---------------------- | --------------------------------- | -------- |
| 1    | gru-minion-handbook-v2 | Moon Door                         | 0.362    |
| 2    | girls-house-rules-v1   | Lab and Living Room / Emergencies | 0.661    |
| 3    | nefario-lab-safety-v1  | Lab Access                        | 0.663    |


v1 password is gone. v2 says the door is welded and to use the garage ramp. The question asks for a password; v2 never states one. Refuse is the literal `data-quality.v1` reading. A useful latest answer would have been: no password; the door is welded; use the garage ramp.

## Notes

- Latest filter is doing the intended work: stop v1 `7:30` and `PURPLE-KAZOO-9` from being the only source on a current-fact question.
- Briefing answers show that clash. Moon-door answers do not. So there is an imbalance where we cannot rely on retrieving chunks from documents with different versions.
- Copy `runs/data-quality-20260925T233948Z.json` to `docs/data-quality-run.json` if you want the payload next to this file, same as the mini-corpus and eval runs.

