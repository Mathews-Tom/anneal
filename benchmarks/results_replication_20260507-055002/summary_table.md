# Benchmark Analysis Summary

## Statistical Comparisons

| Target | Comparison | Mean A | Mean B | p (raw) | p (corrected) | Cohen's d [95% CI] | Significant |
|--------|------------|--------|--------|---------|---------------|--------------------|-------------|
| B3 | control vs treatment | 20000.6000 | 40000.2000 | 0.257 | 0.514 | 0.43 [-0.28, 1.28] | No |
| B5 | control vs treatment | 5.0667 | 5.0667 | 1.000 | 1.000 | 0.00 [-0.78, 0.74] | No |
| B3 | greedy vs treatment | 26667.1333 | 40000.2000 | 0.414 | 0.828 | 0.28 [-0.43, 1.07] | No |
| B5 | greedy vs treatment | 5.0000 | 5.0667 | 0.964 | 0.964 | 0.07 [-0.68, 0.79] | No |
| B3 | greedy vs control | 26667.1333 | 20000.6000 | 0.705 | 1.000 | -0.15 [-0.97, 0.55] | No |
| B5 | greedy vs control | 5.0000 | 5.0667 | 0.856 | 1.000 | 0.06 [-0.70, 0.79] | No |


## Descriptive Statistics

| Target | Config | Metric | N | Mean | Median | Std | IQR | Min | Max |
|--------|--------|--------|---|------|--------|-----|-----|-----|-----|
| B3 | control | acceptance_rate | 15 | 0.4439 | 0.4615 | 0.1149 | 0.1159 | 0.1765 | 0.6316 |
| B3 | control | convergence_experiment | 15 | 2.7333 | 2.0000 | 1.5337 | 1.0000 | 1.0000 | 7.0000 |
| B3 | control | final_score | 15 | 20000.6000 | 1.0000 | 41403.1055 | 0.0000 | 1.0000 | 99999.0000 |
| B3 | control | total_cost_usd | 15 | 13.4494 | 12.5733 | 4.2102 | 4.8634 | 7.5690 | 25.1352 |
| B3 | greedy | acceptance_rate | 15 | 0.1649 | 0.1290 | 0.0973 | 0.1531 | 0.0513 | 0.3333 |
| B3 | greedy | convergence_experiment | 15 | 2.8667 | 3.0000 | 1.4075 | 1.5000 | 1.0000 | 6.0000 |
| B3 | greedy | final_score | 15 | 26667.1333 | 1.0000 | 45772.8553 | 49999.0000 | 1.0000 | 99999.0000 |
| B3 | greedy | total_cost_usd | 15 | 17.7425 | 14.6524 | 7.9045 | 12.9752 | 6.8392 | 32.1652 |
| B3 | treatment | acceptance_rate | 15 | 0.3909 | 0.4286 | 0.1007 | 0.1314 | 0.2000 | 0.5333 |
| B3 | treatment | convergence_experiment | 15 | 3.1333 | 3.0000 | 1.9952 | 1.5000 | 1.0000 | 8.0000 |
| B3 | treatment | final_score | 15 | 40000.2000 | 1.0000 | 50708.2411 | 99998.0000 | 1.0000 | 99999.0000 |
| B3 | treatment | total_cost_usd | 15 | 12.4517 | 12.0508 | 2.6307 | 3.2510 | 7.7095 | 16.9355 |
| B5 | control | acceptance_rate | 15 | 0.3267 | 0.2500 | 0.1522 | 0.1750 | 0.2000 | 0.7500 |
| B5 | control | convergence_experiment | 15 | 1.5333 | 1.0000 | 0.7432 | 1.0000 | 1.0000 | 3.0000 |
| B5 | control | final_score | 15 | 5.0667 | 5.0000 | 1.0328 | 2.0000 | 3.0000 | 6.0000 |
| B5 | control | total_cost_usd | 15 | 4.3060 | 4.2272 | 0.7451 | 0.7070 | 2.9296 | 5.7042 |
| B5 | greedy | acceptance_rate | 15 | 0.3989 | 0.4000 | 0.1493 | 0.1583 | 0.2000 | 0.7500 |
| B5 | greedy | convergence_experiment | 15 | 2.0667 | 2.0000 | 1.0328 | 1.5000 | 1.0000 | 4.0000 |
| B5 | greedy | final_score | 15 | 5.0000 | 5.0000 | 1.0690 | 1.5000 | 3.0000 | 6.0000 |
| B5 | greedy | total_cost_usd | 15 | 4.2386 | 4.4359 | 0.7547 | 1.0002 | 2.9089 | 5.7838 |
| B5 | treatment | acceptance_rate | 15 | 0.4433 | 0.4000 | 0.1613 | 0.1000 | 0.2000 | 0.8000 |
| B5 | treatment | convergence_experiment | 15 | 2.5333 | 2.0000 | 1.1872 | 1.5000 | 1.0000 | 5.0000 |
| B5 | treatment | final_score | 15 | 5.0667 | 5.0000 | 0.9612 | 2.0000 | 4.0000 | 6.0000 |
| B5 | treatment | total_cost_usd | 15 | 4.3868 | 4.0993 | 0.7056 | 0.9896 | 3.4095 | 5.9255 |


## Treatment-Control Metric Deltas

These rows are descriptive configuration-level deltas. They do not attribute effects to individual enhancements, because the treatment configuration changes multiple mechanisms at once.

| Metric Label | Target | Metric | Direction | Control Mean | Treatment Mean | Relative Change (%) |
|-------------|--------|--------|-----------|--------------|----------------|---------------------|
| Final score | B3 | final_score | minimize | 20000.6000 | 40000.2000 | -100.00% |
| Final score | B5 | final_score | maximize | 5.0667 | 5.0667 | +0.00% |
| Total cost | B3 | total_cost_usd | minimize | 13.4494 | 12.4517 | +7.42% |
| Total cost | B5 | total_cost_usd | maximize | 4.3060 | 4.3868 | +1.88% |
| Acceptance rate | B3 | acceptance_rate | minimize | 0.4439 | 0.3909 | +11.93% |
| Acceptance rate | B5 | acceptance_rate | maximize | 0.3267 | 0.4433 | +35.71% |
| Convergence experiment | B3 | convergence_experiment | minimize | 2.7333 | 3.1333 | -14.63% |
| Convergence experiment | B5 | convergence_experiment | maximize | 1.5333 | 2.5333 | +65.22% |
