# Results: visa_validity — Германия

## Total records

100 messages processed (out of 444 relevant messages in the dataset).

Tag distribution:
- #одобрено: 92
- #одобрение: 2
- #отказ: 6

## Distribution of extracted fields

### validity_days

Non-null values (41 records total):

| Days | Count | Notes |
|------|-------|-------|
| 5    | 4     | Short business/travel trips |
| 6    | 4     | |
| 7    | 6     | One week |
| 10   | 1     | |
| 12   | 3     | |
| 13   | 1     | |
| 14   | 4     | Two weeks |
| 15   | 2     | |
| 16   | 1     | |
| 17   | 2     | |
| 20   | 1     | |
| 27   | 1     | |
| 28   | 1     | |
| 30   | 2     | One month; one is a post-appeal visa under #отказ tag |
| 36   | 1     | |
| 44   | 1     | |
| 60   | 1     | |
| 90   | 2     | Three months (national visa / business) |
| 365  | 3     | One year (IT specialist / Chancenkarte / student national visas) |

Null (no validity days mentioned): 59 records.

### exactly_under_dates

| Value | Count |
|-------|-------|
| true  | 72    |
| false | 12    |
| null  | 16    |

## Quality assessment

### Correct extractions (spot-check of 10 records)

All 10 spot-checked records were correctly extracted:

- Short approved visas with explicit day counts (e.g. "6 дней одноразовая под даты") → correct day count and exactly_under_dates=true.
- Messages mentioning only "под даты" without a day count → validity_days=null, exactly_under_dates=true.
- Rejections with no visa granted → both null.
- Longer visas: "на 3 месяца" → 90, "дали на год" → 365.

### Ambiguous cases

- 44 records have exactly_under_dates=true but validity_days=null. This is expected and correct: many messages say "дали под даты" / "ровно под поездку" without stating the explicit number of days, so the duration is unresolvable from text alone.
- 16 records have both fields null for approved messages: texts lack any validity or date-match information and only describe the application process.

### Anomalies

- One #отказ record (ID 61958) has validity_days=30 and exactly_under_dates=false. Correct: the message describes an initial rejection followed by a successful appeal resulting in a multi-visa for one month. The tag reflects the original event but the text contains the final positive outcome.
- One record (ID 187979, 90 days) has exactly_under_dates=true: "дали мульти три месяца 15 дней пребывания (под даты броней гостиницы)". Extraction is correct — visa validity matched booked hotel dates.

## Prompt iterations

1 round. The initial prompt was sufficient; all 5 sample messages were extracted correctly on the first run without requiring any prompt edits.
