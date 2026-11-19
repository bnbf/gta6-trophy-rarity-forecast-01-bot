# gta6-trophy-rarity-forecast-01-bot

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bnbf/gta6-trophy-rarity-forecast-01-bot/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bnbf/gta6-trophy-rarity-forecast-01-bot/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each of the 55 trophies in week one, as 5 quantiles per trophy. Exophase publishes rates as of its own last refresh, hours apart, so a read is a stale snapshot: when the history is flat the model assumes a refresh is due and dilutes each rate by about six (the documented first-refresh jump, 638 → 3,868 players); when the history moves it fits a robust Theil-Sen growth line on logit(rate) against log-time over the last 24 hours of reads and extrapolates to the target, with a spread from the prediction's standard error plus refresh and horizon terms. Practice loss 0.16 on both dev cases vs the 1.51 house baselines.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=bsqaxrmg&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
