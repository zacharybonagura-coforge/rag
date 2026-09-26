# Mini-corpus RAG run

Results from `src/run_mini_corpus.py` on **2026-09-23 23:24:20 UTC**
(`runs/20260923T232420Z.json`).

## Setup

| Piece      | Value                                                    |
| ---------- | -------------------------------------------------------- |
| Corpus     | `data/corpus-tiny/harbor-bike-shop-handbook.md` (v1.0) |
| Chunks     | Hours (section 1), Rentals (section 2)                   |
| Embeddings | `sentence-transformers/all-mpnet-base-v2`              |
| Store      | pgvector, cosine distance (lower is closer)              |
| Generate   | Ollama`mistral:7b`, prompt `mini.v1`                 |
| k          | 2                                                     |

### Handbook

**Hours.** The shop is open Tuesday through Saturday from 10am to 6pm. We are closed Sunday and Monday.

**Rentals.** City bikes rent for $25 a day. Electric bikes rent for $45 a day. A photo ID is required.

## Summary

All six questions cited the intended section. Answers are grounded; some omit extra facts that were in context (Monday closed, 10am open).

| # | Query                                | Citation | Top distance | Answer                                     |
| - | ------------------------------------ | -------- | ------------ | ------------------------------------------ |
| 1 | What days is the shop open?          | Hours    | 0.191        | The shop is open Tuesday through Saturday. |
| 2 | Are you open on Sunday?              | Hours    | 0.286        | No, we are closed on Sunday.               |
| 3 | What time do you close?              | Hours    | 0.396        | The shop closes at 6pm.                    |
| 4 | How much is a city bike?             | Rentals  | 0.265        | A city bike rents for $25 a day.           |
| 5 | How much does an electric bike cost? | Rentals  | 0.347        | An electric bike costs $45 a day.          |
| 6 | Do I need a photo ID to rent?        | Rentals  | 0.695        | Yes, you need a photo ID to rent.          |

## Per-query results

### 1. What days is the shop open?

**Answer:** The shop is open Tuesday through Saturday.

**Citation:** harbor-bike-shop-handbook v1.0 §1 Hours

| Rank | Section | Distance |
| ---- | ------- | -------- |
| 1    | Hours   | 0.191    |
| 2    | Rentals | 0.770    |

Correct. Omits 10am–6pm.

### 2. Are you open on Sunday?

**Answer:** No, we are closed on Sunday.

**Citation:** harbor-bike-shop-handbook v1.0 §1 Hours

| Rank | Section | Distance |
| ---- | ------- | -------- |
| 1    | Hours   | 0.286    |
| 2    | Rentals | 0.859    |

Correct.

### 3. What time do you close?

**Answer:** The shop closes at 6pm.

**Citation:** harbor-bike-shop-handbook v1.0 §1 Hours

| Rank | Section | Distance |
| ---- | ------- | -------- |
| 1    | Hours   | 0.396    |
| 2    | Rentals | 0.879    |

Correct. Omits 10am open.

### 4. How much is a city bike?

**Answer:** A city bike rents for $25 a day.

**Citation:** harbor-bike-shop-handbook v1.0 §2 Rentals

| Rank | Section | Distance |
| ---- | ------- | -------- |
| 1    | Rentals | 0.265    |
| 2    | Hours   | 0.858    |

Correct.

### 5. How much does an electric bike cost?

**Answer:** An electric bike costs $45 a day.

**Citation:** harbor-bike-shop-handbook v1.0 §2 Rentals

| Rank | Section | Distance |
| ---- | ------- | -------- |
| 1    | Rentals | 0.347    |
| 2    | Hours   | 0.894    |

Correct.

### 6. Do I need a photo ID to rent?

**Answer:** Yes, you need a photo ID to rent.

**Citation:** harbor-bike-shop-handbook v1.0 §2 Rentals

| Rank | Section | Distance |
| ---- | ------- | -------- |
| 1    | Rentals | 0.695    |
| 2    | Hours   | 0.969    |

Correct. Weakest retrieve of the six, still ranked Rentals first.
