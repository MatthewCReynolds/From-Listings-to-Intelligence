# Project phases: what we build, what we deliver, how we measure it

Planning document for the Accolade capstone. It is based on Accolade's *Draft Project Plan*
(section numbers below, like "plan 4.1", point to it) and on the synthetic dataset in
`data/realistic-v2/`. Written 2026-10-22.

## 1. The project in one table

Accolade is building a rent-pricing product with two parts, built in order:

| | **Phase 1: Market rent estimate** | **Phase 2: Dynamic price recommendation** |
|---|---|---|
| Question | What are comparable units asking right now for this unit type? | What price should this property post today to reach its occupancy target? |
| Data it needs | Market listings only | The property's own PMS data: leads, applications, leases |
| Main output | Asking rent per unit type: an estimate, a range, and the reasons | Recommended price per unit group per day, the reasons, and which limits applied |
| Also includes | Comparable-property (comp) selection | Concession extraction, net effective rent, 30/60-day occupancy forecast |
| Plan sections | 3, 4.1 (Phase 1) | 2, 4.1 (Phase 2), 4.4, Appendix A |

Phase 2 builds on Phase 1: the recommendation uses the comps and the market rent estimate as anchors.

A **unit type** is a property's bedrooms + bathrooms + a square-footage band (plan, "At what level").
A property-level number is only ever a roll-up, never the thing we model.

## 2. Working with synthetic data: what we can and cannot claim

We don't have the Dwellsy data. The plan (6.6) says synthetic data is for building and testing
the pipeline, and must never be used to choose or tune a model.

That sounds like we are working on nothing, but the synthetic data has one thing real data
never will: an **answer key**. The generator saved the true values behind what we observe, in `truth/`:

| What we estimate | What we observe (input) | Answer key |
|---|---|---|
| Market rent per floor plan | `market.price_observation` (asking rents) | `truth.floorplan_market_rent_monthly` |
| Comp sets | `market.building` | `truth.comp_set` (per-feature scores, rank, selected), `market.report` (stands in for analyst-curated sets) |
| Concession terms | `market.concession_event.specials` (free text) | `truth.concession` |
| Demand funnel and seasonality | `pms.guestcard`, `pms.guestcard_event`, `pms.lease` | `truth.demand_params` |
| The recommender's output | everything in `pms/` | `pms.price_recommendation` + `pms.price_recommendation_details` (the reference system's prices and full audit trail) |
| Duplicate / linked buildings | `market.building_alias`, `pms.property` | the shared `property_id` |
| Edge cases | | `truth.scenario_log`: lease-up, missing attributes, vacancy spikes, rent control |

So our main result is a **methodology result**: for each method, *does it recover the known
truth, and does it beat the plan's baseline?* We can only test that on synthetic data.

| We can claim | We cannot claim |
|---|---|
| The pipeline runs end to end, from raw tables to a price with a range and reasons | Real-world accuracy |
| Method X recovers the true value to within Y% when the data behaves as the generator assumes | The final feature list |
| Method X beats (or doesn't beat) the plan's baseline | Final hyperparameters or model choice |
| Where methods break: lease-up, missing attributes, rent control | |
| Re-running on Dwellsy data means changing a data path, not rewriting code | |

One caution for the presentation: a method can look good just because it matches how the generator
was built. For example, gradient boosting may recover a rent formula that the generator itself
built from the same building attributes. Say this openly; it is part of an honest methodology.

## 3. Rules for both phases

1. **Only use what is known at prediction time.** In Phase 1 that is the day the unit is listed, so
   the listing's later prices, its own days on market, and its price-change counts are not inputs.
2. **`truth_*` and `latent` columns are answer keys, never inputs.**
3. **Never add customer (PMS) data onto market data** (plan 1.3). Phase 1 inputs come from `market/`
   only. Phase 2 may add market data onto PMS data, not the other way round.
4. **Test on what the model hasn't seen:** unseen buildings (`GroupKFold` by `property_id`) and a later
   time period (train on earlier months, test on the last ~6 months).
5. **Report a baseline next to every model.**
6. **Every price carries a range and its reasons** (plan, "What output is expected").

## 4. Phase 1: Market rent estimate

### 4.1 Deliverable

One row per unit type per month:

| Field | Meaning |
|---|---|
| `property_id`, `bed`, `bath`, `sqft_band`, `market`, `month` | The unit type and when |
| `asking_rent_estimate` | Point estimate |
| `range_low`, `range_high` | For example the 10th and 90th percentiles |
| `evidence_count` | Number of comps or listings behind the estimate |
| `comps` | Top comps, each with its similarity and per-feature scores |
| `reasons` | 3–5 plain-language drivers ("larger than comps: +$120") |
| `model_version` | |

When a unit type has no comp evidence, output nothing rather than a default (plan 4.1).

### 4.2 Steps

**Step 1.0: Data foundation (mostly done)**

Done so far:

| Work | File |
|---|---|
| Master table, one row per listing, 967,661 × 378 | `preprocessing/build_master.py` |
| Standardized column types | `preprocessing/standardize_types.py` |
| 59 join checks, all passing | `notebooks/phase_01_market_rent/01_master_table_check.ipynb` |
| Completeness and type of all 378 variables | `notebooks/phase_01_market_rent/02_feature_funnel.ipynb` |
| Feature scoring: Gini, gradient boosting, Lasso | `notebooks/phase_01_market_rent/03_feature_selection_numerical.ipynb` |
| Correlations and redundant pairs, Excel report | `notebooks/phase_01_market_rent/04_numerical_correlations.ipynb` |
| Decision table for all 108 numerical variables (scores, correlations, suggested and user decisions), profile and plots of the kept ones | `notebooks/phase_01_market_rent/05_numerical_final_selection.ipynb` |
| Numeric feature analysis | `preprocessing/analyze_numeric.py` (branch `shikha-edits`) |

Still to fix before modeling:

- **Target.** Change `y` from `truth_lease_event__true_rent` (simulator truth, which Dwellsy won't
  have) to the listing's asking rent when it was listed: `market_price_observation__first_asking_rent`.
  The other 13 `leaky` price columns stay out of X.
- **Answer key.** Keep `truth_floorplan_market_rent__true_market_rent_at_enter` to check results against.
- **Inputs.** Drop `pms_property__*` from X (rule 3). Keep only features known when the unit is listed (rule 1).
- **Re-run** `eda/03` and `eda/04` with these changes. Add `market_building__year_built` back: the
  plan weights vintage at 0.10, and gradient boosting and Lasso both kept it.

Output: the final feature list, with each feature's completeness, Gini, gradient-boosting importance,
Lasso coefficient, and any redundant partner.

**Step 1.1: Comp selection (plan 3.1–3.4)**

1. **Hard filters:** distance, unit count, year built, stories, property class, product type
   (student, senior, affordable vs market-rate), lease-up vs stabilized, building form, rent control.
   A missing value excludes a candidate only when that filter is actually set.
2. **Similarity score** with the plan's baseline weights:

   | Feature | Weight | Scored as |
   |---|---|---|
   | Location | 0.35 | 1 − distance / 5 miles |
   | Interior quality | 0.25 | 1 − difference |
   | Size | 0.15 | 1 − difference / 300 units |
   | Vintage | 0.10 | 1 − difference / 50 years |
   | Amenities | 0.15 | Jaccard overlap |

   Quality scores are empty in this data (`market.quality_scores` has 0 rows), so the plan's rule
   applies: a missing feature's weight is spread over the others.
3. **Keep the top 10** (plan default; maximum 100).
4. **Match floor plans** inside the comp set by bedrooms, bathrooms and sq-ft band.

Check it against `truth.comp_set` and `market.report`: precision and recall of our top 10.
Stretch goal: a learned embedding, which must beat the weighted score (plan 3.1).

**Step 1.2: Baseline rent (plan 4.1)**

Similarity-weighted mean of comp rent per sq ft × the subject's sq ft. This is the minimum bar:
every model must be compared against it.

**Step 1.3: Model**

- Gradient boosting (LightGBM or scikit-learn's `HistGradientBoostingRegressor`) on the step-1.0
  features, plus comp features from step 1.1 (for example the comps' median rent per sq ft).
- A neural network is optional, and only worth it if it beats gradient boosting.
- **Range:** quantile models for the 10th and 90th percentiles, or conformal prediction.
  Then check that about 80% of true values fall inside the range.

**Step 1.4: Explanations**

SHAP values give each feature's contribution to a prediction. Turn the top 3–5 into plain-language
reasons, and show them next to the comp list.

**Step 1.5: Evaluation (plan, "How will usefulness be evaluated")**

| Metric | Compared against |
|---|---|
| MdAPE by unit type and market | Observed asking rent, on held-out buildings and months |
| Interval coverage and width | Target: ~80% coverage for a 10th–90th range |
| MdAPE against true market rent | `truth.floorplan_market_rent_monthly` (synthetic only) |
| Comp precision and recall at 10 | `truth.comp_set`, `market.report` |
| All of the above for the baseline | Step 1.2 |

Break results down by edge case from `truth.scenario_log`: lease-up buildings, buildings with
masked attributes, rent-controlled buildings (DC).

**Step 1.6: Optional**

- **12-month rent forecast** per market (plan 4.4). Start with the plan's minimum baseline:
  a drift model with ±1.96σ intervals.
- **Time to lease.** Survival model in `model/survival_quick.py` (marked unvalidated).
- **Entity resolution** (plan 1.2): a pairwise "same building?" classifier on name/address pairs.
  `market.building_alias` gives labeled duplicates. `pms.property` and `pms.comp_property` carry
  the true `property_id`, so they give labels for linking a customer's property to its market record.

### 4.3 What to present for Phase 1

- Pipeline diagram and data checks: 59 join checks, then the feature funnel (378 → … → final list)
- Comp method with one worked example: a subject building and its 10 comps with their scores
- Results table: baseline vs gradient boosting (vs neural net), with MdAPE and coverage by market and bedrooms
- How close each method gets to the true market rent
- Limitations and what changes when Dwellsy data arrives

## 5. Phase 2: Dynamic price recommendation and net effective rent

The synthetic PMS layer covers 150 properties (Chicago 70, DFW 50, DC 30).

### 5.1 Deliverables

1. **Price recommendation** per property × unit group × day: current price, recommended price,
   demand gap (recent vs needed leads per day), the guardrails that applied, and the reasons.
2. **Concession extraction** per special (plan 2.3): type, value amount, value basis, lease term,
   amortized monthly value, confidence, model version, provenance.
3. **Net effective rent** per unit type = asking rent − amortized monthly concession.
4. **Occupancy forecast** at 30 and 60 days.

### 5.2 Steps

**Step 2.1: Concession extraction (plan 2)**

This is the most self-contained piece, and the labels already exist, so it is a good place to start.

- **Input:** 91,156 specials texts in `market.concession_event`. **Labels:** `truth.concession`.
- **Methods:** a regex/rules baseline vs an LLM extractor. Pin the LLM version and save its
  responses so results are reproducible.
- **The plan's rules:** weeks become months at 7/30; "up to N" means N; never infer an amount
  that isn't stated; keep "extracted, no match" (`unknown`) separate from "never extracted".
- **Flags for review:** more than 3 months free, amortized value above asking rent, unclassified
  text, missing lease term.
- **Evaluate:** type accuracy and value accuracy (a value counts as correct only when the type is
  also correct), by market and by type. Plan reference: a regex baseline gets 92% type and 88% value
  accuracy on 50 synthetic examples.
- **Caution:** the texts were written by an LLM during generation (column `text_style` in
  `truth.concession`). Rules can end up fitting that LLM's phrasing, so report results by `text_style`.

**Step 2.2: Net effective rent (plan 2.4)**

```
amortized_monthly = total_concession_value / lease_term_months   (default term: 12)
  months free:      total_value = months_free × asking_rent
  recurring:        already monthly
effective_rent    = asking_rent − amortized_monthly
```

If asking rent is unknown, a months-free concession has an unknown value, not zero.
Check against `truth.concession.net_effective_rent` and `truth.lease_event.true_effective_rent`.

**Step 2.3: Demand-conversion recommendation (plan 4.1 Phase 2, Appendix A)**

The idea is to run the leasing funnel backwards. Work out how many leads the property needs to
reach its occupancy target, compare that with the leads it's getting, and move the price to close the gap.

| # | Step | Synthetic tables |
|---|---|---|
| 1 | **Leases needed:** from target occupancy, projected occupancy and the target timeline (default 45 days); smooth monthly target steps over 28 days | `pms.occupancy_target`, `pms.property_settings`, `pms.lease`, `pms.unit` |
| 2 | **Leads needed:** divide by the funnel ratios (lead → application → lease → move-in), computed per property over moving windows | `pms.guestcard`, `pms.guestcard_event`, `pms.lease` |
| 3 | **Leads expected:** a seasonal demand model of leads per week (baseline: a 12-month seasonal curve) | the above + `pms.daily_weather`, `pms.weather_alert` |
| 4 | **Bound the target:** clamp it to a band around the seasonal prediction, using the property's own deviation factors | |
| 5 | **Price the gap:** convert the gap between recent and target leads into a price change, using comp price sensitivity | Phase 1 comps, `pms.comp_*` |
| 6 | **Compress and apply guardrails, in order:** daily cap, weekly cap, comp-based floor, ceiling vs recent leases, nudge toward market rent, consistency with demand | `pms.property_settings` |
| | **Explanation:** plain-language reasons, including which guardrail limited the price | |

- **Answer keys:** `truth.demand_params` holds the true seasonal curve, funnel probabilities,
  price sensitivity, deviation bands and weather factor.
- **Reference outputs:** `pms.price_recommendation` and `pms.price_recommendation_details` are
  the reference system's daily prices with every intermediate number. Compare ours to them step by step.
- **Open design work** (the plan calls it "the main modeling work in Phase 2"): the inputs and
  architecture of each model, the formula that turns the gap into a price, and the compression formula.
- **Directions the plan suggests:**
  - partial pooling (hierarchical models) for properties with little data;
  - proper price-elasticity estimation (price and demand are set together, so a plain regression is biased);
  - intervals on the recommended price.

**Step 2.4: Short-horizon forecasting (plan 4.4)**

- **Occupancy** at 30 and 60 days: today's occupied units − move-outs + move-ins, from `pms.lease`.
- **Renewal rate** from `pms.renewal_offer`.
- **Monthly early-termination rate**, shrunk toward a low prior.

**Step 2.5: Evaluation (plan Appendix A, "How to evaluate it")**

| Piece | Metric | Compared against |
|---|---|---|
| Demand forecast | Error on leads per week | Seasonal baseline |
| Funnel ratios | Calibration: predicted vs realized | `truth.demand_params` |
| Recommendation | Agreement with the reference; direction matches the demand gap; how often each guardrail binds | `pms.price_recommendation(_details)` |
| Concession extraction | Type and value accuracy | `truth.concession` |
| Occupancy forecast | Error at 30 and 60 days | Held-out months |

A limitation to state openly: the plan's real test, "would following our price have moved occupancy
toward target?", needs a counterfactual. With fixed data we can only compare against the reference
recommender and check direction, not measure what would have happened.

### 5.3 What to present for Phase 2

- Concession extraction: regex vs LLM accuracy table, with examples of each one's mistakes
- The demand-conversion method walked through for one property, with the number at every step
- Our recommendation vs the reference over time, and how often each guardrail applies
- How well we recover the true seasonal curve and funnel ratios

## 6. Out of scope

**Computer vision / interior quality (plan 5):** `market.photo` and `market.quality_scores` have no
rows in this dataset, so there is nothing to build on. Mention it as future work only.

## 7. Suggested order

1. **Fix the foundation** (step 1.0). It's small, and everything else depends on it.
2. **In parallel:** comps + baseline (1.1–1.2), model + ranges + explanations (1.3–1.4),
   concession extraction (2.1).
3. **Then:** Phase 1 evaluation (1.5) and net effective rent (2.2).
4. **Then:** demand-conversion recommendation (2.3) and occupancy forecast (2.4).
5. **Optional items last** (1.6).

**Minimum for the final presentation:**
- Phase 1 complete: baseline, model, ranges, evaluation.
- For Phase 2: concession extraction, plus one end-to-end walk-through of the demand-conversion method.

## 8. Questions to take to Accolade

Still open on their side (plan, "Open items"):
- the date of the first Dwellsy delivery;
- whether the history is daily or weekly;
- whether the feed carries concession text and photos;
- the PMS schemas and the 120-field catalog.

Ours:
- Can recovery-against-truth be the main evaluation in our capstone, given there is no real data yet?
- How wide should a unit type's sq-ft band be?
- What range do they want: 80%, 90%?
- In Phase 2, should we replicate the reference recommender (`pms.price_recommendation`) or try to improve on it?

## Glossary

| Term | Meaning |
|---|---|
| Asking rent | The price on the listing |
| Effective rent | Asking rent minus concessions, spread per month |
| Comp | A comparable property |
| MdAPE | Median absolute percentage error: the median of \|predicted − actual\| / actual |
| Interval coverage | Share of actual values that fall inside the predicted range |
| `GroupKFold` by property | Each building is entirely in training or entirely in testing, never both |
| Guardrail | A limit on how far a recommended price can move |
| SHAP | A method that splits one prediction into each feature's contribution |
