# Business interpretation

## Business question

Which telecom customers should a retention team **review** for voluntary
contact three months before a potential churn event?

## Analytical framing

A ranking model prioritizes accounts. Classification performance measures
how well risk signals predict churn, **not** whether an offer prevents it.

Useful pilot measures include:
- Number of customers flagged and fraction truly churning.
- Contact capacity, outreach cost and offer redemption.
- Calibration of risk and precision by operational segment.
- Incremental retention from a **randomized** experiment.
- Net customer contribution margin and fairness of outreach/error rates.

## Scenario formula (illustrative, not inferred from this dataset)

```text
Expected incremental profit =
    contacted_accounts × true_incremental_retention_uplift × contribution_margin
  − total_outreach_and_offer_costs
```

The UCI data contains no randomized intervention and cannot estimate the
uplift term. Avoid presenting classification F2 as business ROI.
