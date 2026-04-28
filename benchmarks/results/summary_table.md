# Benchmark Analysis Summary

## Statistical Comparisons

| Target | Comparison | Mean A | Mean B | p (raw) | p (corrected) | Cohen's d [95% CI] | Significant |
|--------|------------|--------|--------|---------|---------------|--------------------|-------------|
| B1 | control vs treatment | 2.8600 | 2.9600 | 0.625 | 1.000 | 0.18 [-1.25, 2.26] | No |
| B2 | control vs treatment | 2.5400 | 2.8000 | 0.250 | 1.000 | 0.41 [-1.25, 1.96] | No |
| B3 | control vs treatment | 49.6450 | 40.8800 | 1.000 | 1.000 | -0.17 [-1.08, 0.78] | No |
| B4 | control vs treatment | 0.7010 | 0.7832 | 0.250 | 1.000 | 0.51 [-0.75, 2.25] | No |
| B5 | control vs treatment | 1.2400 | 1.5500 | 0.230 | 1.000 | 0.57 [-0.29, 1.55] | No |
| B1 | greedy vs treatment | 2.8000 | 2.9600 | 0.250 | 1.000 | 0.26 [-1.09, 2.13] | No |
| B2 | greedy vs treatment | 2.6800 | 2.8000 | 0.750 | 1.000 | 0.21 [-1.24, 1.76] | No |
| B3 | greedy vs treatment | 55.8030 | 40.8800 | 0.432 | 1.000 | -0.26 [-1.15, 0.69] | No |
| B4 | greedy vs treatment | 0.7370 | 0.7832 | 0.625 | 1.000 | 0.30 [-1.09, 1.74] | No |
| B5 | greedy vs treatment | 1.0300 | 1.5500 | 0.023 | 0.117 | 1.13 [0.36, 2.21] | No |
| B1 | raw vs treatment | 0.0000 | 2.9600 | 0.062 | 0.250 | 6.59 [5.40, 24.96] | No |
| B2 | raw vs treatment | 0.0000 | 2.8000 | 0.062 | 0.250 | 8.00 [6.18, 21.98] | No |
| B3 | raw vs treatment | 187.2320 | 40.8800 | 0.557 | 0.557 | -0.84 [-1.83, -0.06] | No |
| B4 | raw vs treatment | 0.1740 | 0.7832 | 0.062 | 0.250 | 3.20 [2.64, 13.33] | No |
| B5 | raw vs treatment | 0.0000 | 1.5500 | 0.002 | 0.010 | 3.93 [2.97, 7.55] | Yes |
| B1 | raw vs control | 0.0000 | 2.8600 | 0.062 | 0.250 | 9.21 [7.18, 20.60] | No |
| B2 | raw vs control | 0.0000 | 2.5400 | 0.062 | 0.250 | 4.83 [3.74, 26.85] | No |
| B3 | raw vs control | 187.2320 | 49.6450 | 0.557 | 0.557 | -0.78 [-1.78, 0.02] | No |
| B4 | raw vs control | 0.1740 | 0.7010 | 0.062 | 0.250 | 2.43 [1.59, 7.24] | No |
| B5 | raw vs control | 0.0000 | 1.2400 | 0.002 | 0.010 | 3.30 [2.30, 5.86] | Yes |
| B1 | greedy vs control | 2.8000 | 2.8600 | 0.750 | 1.000 | 0.12 [-1.38, 1.76] | No |
| B2 | greedy vs control | 2.6800 | 2.5400 | 0.875 | 1.000 | -0.20 [-1.68, 1.36] | No |
| B3 | greedy vs control | 55.8030 | 49.6450 | 0.322 | 1.000 | -0.10 [-1.03, 0.86] | No |
| B4 | greedy vs control | 0.7370 | 0.7010 | 0.625 | 1.000 | -0.19 [-1.90, 1.12] | No |
| B5 | greedy vs control | 1.0300 | 1.2400 | 0.477 | 1.000 | 0.47 [-0.39, 1.48] | No |


## Descriptive Statistics

| Target | Config | Metric | N | Mean | Median | Std | IQR | Min | Max |
|--------|--------|--------|---|------|--------|-----|-----|-----|-----|
| B1 | control | acceptance_rate | 5 | 0.6601 | 0.6471 | 0.0286 | 0.0400 | 0.6333 | 0.7000 |
| B1 | control | convergence_experiment | 5 | 3.6000 | 2.0000 | 3.2094 | 5.0000 | 1.0000 | 8.0000 |
| B1 | control | final_score | 5 | 2.8600 | 2.9000 | 0.4393 | 0.5000 | 2.4000 | 3.5000 |
| B1 | control | total_cost_usd | 5 | 2.1879 | 0.3326 | 4.2173 | 0.3608 | 0.0827 | 9.7251 |
| B1 | greedy | acceptance_rate | 5 | 0.0320 | 0.0400 | 0.0110 | 0.0200 | 0.0200 | 0.0400 |
| B1 | greedy | convergence_experiment | 5 | 3.4000 | 2.0000 | 3.0496 | 4.0000 | 1.0000 | 8.0000 |
| B1 | greedy | final_score | 5 | 2.8000 | 2.8000 | 0.5788 | 0.8000 | 2.2000 | 3.6000 |
| B1 | greedy | total_cost_usd | 5 | 2.5920 | 0.4830 | 5.0481 | 0.3162 | 0.1715 | 11.6178 |
| B1 | raw | acceptance_rate | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B1 | raw | convergence_experiment | 5 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| B1 | raw | final_score | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B1 | raw | total_cost_usd | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B1 | treatment | acceptance_rate | 5 | 0.6181 | 0.6250 | 0.0590 | 0.0322 | 0.5200 | 0.6735 |
| B1 | treatment | convergence_experiment | 5 | 4.0000 | 2.0000 | 4.2426 | 4.0000 | 1.0000 | 11.0000 |
| B1 | treatment | final_score | 5 | 2.9600 | 3.1000 | 0.6348 | 1.1000 | 2.2000 | 3.6000 |
| B1 | treatment | total_cost_usd | 5 | 2.7222 | 0.4788 | 5.2520 | 0.4086 | 0.1867 | 12.1116 |
| B2 | control | acceptance_rate | 5 | 0.5207 | 0.5000 | 0.0419 | 0.0417 | 0.4783 | 0.5833 |
| B2 | control | convergence_experiment | 5 | 4.4000 | 4.0000 | 3.7815 | 5.0000 | 1.0000 | 10.0000 |
| B2 | control | final_score | 5 | 2.5400 | 2.9000 | 0.7436 | 1.1000 | 1.6000 | 3.3000 |
| B2 | control | total_cost_usd | 5 | 3.1211 | 0.9541 | 5.1698 | 0.5046 | 0.5638 | 12.3599 |
| B2 | greedy | acceptance_rate | 5 | 0.0445 | 0.0408 | 0.0087 | 0.0017 | 0.0400 | 0.0600 |
| B2 | greedy | convergence_experiment | 5 | 10.2000 | 2.0000 | 11.4978 | 17.0000 | 2.0000 | 26.0000 |
| B2 | greedy | final_score | 5 | 2.6800 | 2.6000 | 0.6301 | 0.5000 | 1.8000 | 3.5000 |
| B2 | greedy | total_cost_usd | 5 | 2.6356 | 1.1666 | 3.9320 | 0.5794 | 0.5571 | 9.6492 |
| B2 | raw | acceptance_rate | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B2 | raw | convergence_experiment | 5 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| B2 | raw | final_score | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B2 | raw | total_cost_usd | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B2 | treatment | acceptance_rate | 5 | 0.4880 | 0.5000 | 0.0610 | 0.0600 | 0.4000 | 0.5600 |
| B2 | treatment | convergence_experiment | 5 | 16.2000 | 2.0000 | 20.3519 | 30.0000 | 1.0000 | 44.0000 |
| B2 | treatment | final_score | 5 | 2.8000 | 2.8000 | 0.4950 | 0.5000 | 2.2000 | 3.5000 |
| B2 | treatment | total_cost_usd | 5 | 2.8586 | 0.7876 | 4.7515 | 0.1642 | 0.6531 | 11.3573 |
| B3 | control | acceptance_rate | 10 | 0.6656 | 0.6751 | 0.2199 | 0.3249 | 0.3778 | 1.0000 |
| B3 | control | convergence_experiment | 10 | 6.5000 | 1.5000 | 12.0577 | 3.7500 | 1.0000 | 40.0000 |
| B3 | control | final_score | 10 | 49.6450 | 31.7100 | 56.0553 | 74.2900 | 2.4600 | 173.0700 |
| B3 | control | total_cost_usd | 10 | 2.7080 | 2.3818 | 1.2810 | 1.7028 | 0.6905 | 4.8338 |
| B3 | greedy | acceptance_rate | 10 | 0.2570 | 0.1611 | 0.2946 | 0.1815 | 0.0000 | 1.0000 |
| B3 | greedy | convergence_experiment | 10 | 5.1000 | 2.0000 | 9.2069 | 2.7500 | 1.0000 | 31.0000 |
| B3 | greedy | final_score | 10 | 55.8030 | 15.5250 | 66.7596 | 94.5825 | 2.8600 | 175.3600 |
| B3 | greedy | total_cost_usd | 10 | 2.1803 | 1.8732 | 1.4064 | 1.6575 | 0.3023 | 4.5313 |
| B3 | raw | acceptance_rate | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B3 | raw | convergence_experiment | 10 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| B3 | raw | final_score | 10 | 187.2320 | 0.0000 | 241.8180 | 459.1850 | 0.0000 | 484.2400 |
| B3 | raw | total_cost_usd | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B3 | treatment | acceptance_rate | 10 | 0.6172 | 0.6492 | 0.1703 | 0.1899 | 0.3023 | 0.8333 |
| B3 | treatment | convergence_experiment | 10 | 10.8000 | 2.5000 | 17.2421 | 7.2500 | 1.0000 | 46.0000 |
| B3 | treatment | final_score | 10 | 40.8800 | 9.6800 | 48.3067 | 75.6075 | 2.7700 | 127.3900 |
| B3 | treatment | total_cost_usd | 10 | 2.8448 | 2.6245 | 0.9933 | 1.2657 | 1.3371 | 4.4249 |
| B4 | control | acceptance_rate | 5 | 0.8131 | 0.8250 | 0.0377 | 0.0272 | 0.7500 | 0.8444 |
| B4 | control | convergence_experiment | 5 | 3.2000 | 2.0000 | 3.3466 | 2.0000 | 1.0000 | 9.0000 |
| B4 | control | final_score | 5 | 0.7010 | 0.7000 | 0.1930 | 0.1250 | 0.4800 | 1.0000 |
| B4 | control | total_cost_usd | 5 | 1.5655 | 1.5162 | 0.8825 | 1.4759 | 0.7036 | 2.6874 |
| B4 | greedy | acceptance_rate | 5 | 0.2328 | 0.0417 | 0.4289 | 0.0009 | 0.0400 | 1.0000 |
| B4 | greedy | convergence_experiment | 5 | 5.6000 | 2.0000 | 8.0808 | 1.0000 | 1.0000 | 20.0000 |
| B4 | greedy | final_score | 5 | 0.7370 | 0.7250 | 0.1776 | 0.1000 | 0.5100 | 1.0000 |
| B4 | greedy | total_cost_usd | 5 | 1.2010 | 0.8621 | 0.7642 | 1.1676 | 0.3603 | 2.0900 |
| B4 | raw | acceptance_rate | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B4 | raw | convergence_experiment | 5 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| B4 | raw | final_score | 5 | 0.1740 | 0.0000 | 0.2383 | 0.4350 | 0.0000 | 0.4350 |
| B4 | raw | total_cost_usd | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B4 | treatment | acceptance_rate | 5 | 0.7940 | 0.8049 | 0.0364 | 0.0581 | 0.7500 | 0.8333 |
| B4 | treatment | convergence_experiment | 5 | 14.0000 | 1.0000 | 18.7883 | 24.0000 | 1.0000 | 42.0000 |
| B4 | treatment | final_score | 5 | 0.7832 | 0.7500 | 0.1247 | 0.0660 | 0.7000 | 1.0000 |
| B4 | treatment | total_cost_usd | 5 | 1.4976 | 1.6211 | 0.5196 | 0.3400 | 0.7740 | 2.1897 |
| B5 | control | acceptance_rate | 10 | 0.4242 | 0.4381 | 0.1014 | 0.1278 | 0.2308 | 0.5349 |
| B5 | control | convergence_experiment | 10 | 12.0000 | 8.0000 | 12.1472 | 11.5000 | 1.0000 | 42.0000 |
| B5 | control | final_score | 10 | 1.2400 | 1.1500 | 0.5317 | 0.6750 | 0.3000 | 2.0000 |
| B5 | control | total_cost_usd | 10 | 4.2203 | 1.2668 | 8.6001 | 0.6684 | 0.3877 | 28.1867 |
| B5 | greedy | acceptance_rate | 10 | 0.0452 | 0.0421 | 0.0166 | 0.0112 | 0.0222 | 0.0741 |
| B5 | greedy | convergence_experiment | 10 | 11.7000 | 8.5000 | 10.7295 | 13.2500 | 2.0000 | 36.0000 |
| B5 | greedy | final_score | 10 | 1.0300 | 1.0000 | 0.3368 | 0.3000 | 0.5000 | 1.6000 |
| B5 | greedy | total_cost_usd | 10 | 4.3636 | 1.2720 | 7.7280 | 0.2155 | 0.9103 | 25.2571 |
| B5 | raw | acceptance_rate | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B5 | raw | convergence_experiment | 10 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| B5 | raw | final_score | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B5 | raw | total_cost_usd | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B5 | treatment | acceptance_rate | 10 | 0.3678 | 0.3466 | 0.1552 | 0.2564 | 0.1250 | 0.5556 |
| B5 | treatment | convergence_experiment | 10 | 9.6000 | 7.5000 | 5.7194 | 5.5000 | 3.0000 | 22.0000 |
| B5 | treatment | final_score | 10 | 1.5500 | 1.5000 | 0.5583 | 0.5750 | 0.7000 | 2.6000 |
| B5 | treatment | total_cost_usd | 10 | 3.8003 | 0.7615 | 9.5562 | 0.4280 | 0.3820 | 30.9831 |


## Enhancement Attribution

| Enhancement | Target | Metric | Direction | Control Mean | Treatment Mean | Relative Change (%) |
|-------------|--------|--------|-----------|--------------|----------------|---------------------|
| Strategy Manifest | B1 | final_score | maximize | 2.8600 | 2.9600 | +3.50% |
| Strategy Manifest | B2 | final_score | maximize | 2.5400 | 2.8000 | +10.24% |
| Strategy Manifest | B3 | final_score | minimize | 49.6450 | 40.8800 | +17.66% |
| Strategy Manifest | B4 | final_score | maximize | 0.7010 | 0.7832 | +11.73% |
| Strategy Manifest | B5 | final_score | maximize | 1.2400 | 1.5500 | +25.00% |
| Dual-Agent Mutation | B1 | total_cost_usd | maximize | 2.1879 | 2.7222 | +24.42% |
| Dual-Agent Mutation | B2 | total_cost_usd | maximize | 3.1210 | 2.8586 | -8.41% |
| Dual-Agent Mutation | B3 | total_cost_usd | minimize | 2.7080 | 2.8448 | -5.05% |
| Dual-Agent Mutation | B4 | total_cost_usd | maximize | 1.5655 | 1.4976 | -4.34% |
| Dual-Agent Mutation | B5 | total_cost_usd | maximize | 4.2203 | 3.8003 | -9.95% |
| Two-Phase Mutation | B1 | acceptance_rate | maximize | 0.6601 | 0.6181 | -6.36% |
| Two-Phase Mutation | B2 | acceptance_rate | maximize | 0.5207 | 0.4880 | -6.27% |
| Two-Phase Mutation | B3 | acceptance_rate | minimize | 0.6656 | 0.6172 | +7.26% |
| Two-Phase Mutation | B4 | acceptance_rate | maximize | 0.8131 | 0.7940 | -2.35% |
| Two-Phase Mutation | B5 | acceptance_rate | maximize | 0.4242 | 0.3678 | -13.31% |
| Lineage Context | B1 | convergence_experiment | maximize | 3.6000 | 4.0000 | +11.11% |
| Lineage Context | B2 | convergence_experiment | maximize | 4.4000 | 16.2000 | +268.18% |
| Lineage Context | B3 | convergence_experiment | minimize | 6.5000 | 10.8000 | -66.15% |
| Lineage Context | B4 | convergence_experiment | maximize | 3.2000 | 14.0000 | +337.50% |
| Lineage Context | B5 | convergence_experiment | maximize | 12.0000 | 9.6000 | -20.00% |
| Episodic Memory | B1 | final_score | maximize | 2.8600 | 2.9600 | +3.50% |
| Episodic Memory | B2 | final_score | maximize | 2.5400 | 2.8000 | +10.24% |
| Episodic Memory | B3 | final_score | minimize | 49.6450 | 40.8800 | +17.66% |
| Episodic Memory | B4 | final_score | maximize | 0.7010 | 0.7832 | +11.73% |
| Episodic Memory | B5 | final_score | maximize | 1.2400 | 1.5500 | +25.00% |
