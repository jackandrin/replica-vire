# ReplicaVire Investment Firm

Providing investment opportunities to college students and young adults through innovative machine learning based investment alternatives.

🏆 **Winner of the MLH Best Use of the Snowflake API Track at PalmettoHacks 2026**

## About ReplicaVire

ReplicaVire was created as a start-up proposal for the PalmettoHacks 2026 Hackathon by Jack Andrin, Jojo Diggs, Connor Parks, and Tristan Peyton. The focal point of the proposal was to offer young adults and college students the opportunity to enter the world of investing without needing a large amount of disposable income at the beginning.

## How We Attacked The Problem

We saw how larger investment firms have large barriers of entry for creating an account and to begin trading. This is due to the minimum trade amounts, deposits for opening an account, minimum balance, etc. These types of requirements just aren't feasible for many college students or adults just beginning their careers. Therefore, we looked at ways to reduce these costs.

## What We Proposed

ReplicaVire replaces the human analyst team with machine learning. Language models read news, filings, and reviews and score them, quantitative models add price-based signals, and a trader model uses those signals to decide what to buy, sell, or hold. A rule-based risk layer limits position sizes and losses so no model can trade outside set bounds. Trade data is stored in Tiger Data and analyzed in Snowflake, where Snowflake Cortex AI writes plain-language explanations of each trade for users. Lower research costs mean lower barriers for customers.

This is a hackathon prototype that uses paper trading only. It is not investment advice, and a real launch would require proper registration and licensed partners.

## What We Learned

Building ReplicaVire taught us that designing the reward matters more than choosing the model, since returns have to account for risk and trading costs to mean anything. We also learned that language models can leak knowledge of past events into backtests, which makes live paper trading the honest test. A simple regression baseline proved useful as a check against overfitting and bugs, and we found that safety rules belong outside the models and that trade explanations should never feed back into them.

## How We Look To Expand

Next, we want to run longer live paper-trading tests against benchmarks and compare our current trader with a reinforcement learning approach. We also plan to add richer data sources, such as earnings calls and licensed review data, and to build goal-based portfolios and educational features aimed at young investors. Over the longer term, we would work with licensed partners on compliance, with the aim of running a small real-money pilot.

## Links
_Presentation Slides_
_Live Demo_
