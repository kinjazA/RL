# SFT Data Report

Input: `data\sft_clean_v2.csv`
Rows: 5,180

## Columns

`question`, `answer`, `source`, `domain`, `answer_type`, `normalized_question`, `q_len`, `a_len`

## Source Distribution

```text
source
career_qa             1620
local_interview_qa    1220
general_alpaca        1160
ml_interview           482
general_oasst          368
se_interview           174
ds_qa_treasury         145
hr_interview            11
```

## Domain Distribution

```text
domain
career        1620
general       1528
behavioral    1231
ml             482
se             174
ds             145
```

## Lengths

Question chars: mean=81.1, median=64.0, p90=138.0, p95=157.0, min=14, max=2092
Answer chars: mean=349.3, median=258.0, p90=610.0, p95=697.0, min=50, max=2727

## Duplicate Questions

Repeated normalized questions: 54
Rows inside repeated questions: 158

Top repeated normalized questions:

```text
  3x  What's a professional skill you wish you were better at?
  3x  What would you say is your biggest weakness?
  3x  Where do you think you still have room to grow as a Product Manager?
  3x  Tell me about an area you're actively working to improve.
  3x  What feedback have you received that was hard to hear but useful?
  3x  What's a skill you bring that you're most confident in?
  3x  What strength do you rely on most when things get difficult?
  3x  What would you say is your greatest strength?
  3x  What do you think sets you apart from other Product Manager candidates?
  3x  What would your manager or teammates say you're best at?
  3x  Why do you want to work as a Product Manager?
  3x  What does success look like for you in this role a year from now?
  3x  Why should we hire you for this Product Manager role over other candidates?
  3x  Why are you looking to leave your current role?
  3x  What draws you to the technology and software field?
  3x  What excites you most about this position?
  3x  Where do you see your career in five years?
  3x  What motivates you day-to-day in this line of work?
  3x  Why are you interested in this company specifically?
  3x  What part of being a Product Manager do you find most fulfilling?
```

## Suspicious General OASST Rows

Rows flagged by length heuristics: 0
