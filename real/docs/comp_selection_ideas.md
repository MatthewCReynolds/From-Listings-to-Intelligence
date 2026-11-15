# Comparable sets (comps): how we could build them

Ideas for the team, written 2026-11-14 and updated 2026-11-15: we now work with the real Dwellsy
data only, not the synthetic data. Nothing here is built yet: the repo has no comp code. The plan
references ("plan 3.1") point to Accolade's *Draft Project Plan*, and every number was measured on
the Dwellsy exports for Chicago and DC.

## In short

1. **Build the plan's baseline first.** Hard filters, then a weighted similarity score, then the
   top 10 comps, then matching floor plans inside them. Every improvement has to beat it (plan 3.1).
2. **There is no answer key yet.** Until Accolade sends analyst-selected comp sets, nothing tells
   us which comps are "right". For now, check the code on a few buildings worked out by hand, and
   compare methods by rent (next point).
3. **To compare methods, use rent.** Good comps should predict a building's rent: the comp
   baseline (comps' rent per sq ft × the unit's sq ft) should have a low error on buildings and
   months the method never saw. Then check against Accolade's analyst comp sets when they arrive.
4. **The real data is missing about half the filters** (quality, stories, class, product type,
   lease-up, rent control). Some have reasonable stand-ins; the rest have to be reported as gaps.

## 1. What Accolade asks for

From plan section 3:

- **Two modes.** Automatic: score every nearby candidate and return the top N (default 10, maximum
  100). Curated: a manager can exclude comps, force-include others and set a price adjustment per
  comp (adjusted price = comp price ÷ adjustment).
- **Hard filters** on distance, unit count, year built, stories, property class and special-demand
  proximity (university, hospital). A missing attribute excludes a candidate only when that filter
  is actually set.
- **Exclusion rules, not penalties** (3.3): product type (student, senior, affordable/LIHTC vs
  market-rate), lease-up vs stabilized, class tier (A/B/C), building form (high-rise vs garden),
  rent-controlled status, single-family or scattered-site vs multifamily.
- **Similarity score with a per-feature breakdown**, using these starting weights:

  | Feature | Weight | Scored as |
  |---|---|---|
  | Location | 0.35 | 1 − distance / 5 miles |
  | Interior quality | 0.25 | 1 − difference |
  | Size | 0.15 | 1 − difference / 300 units |
  | Vintage | 0.10 | 1 − difference / 50 years |
  | Amenities | 0.15 | Jaccard overlap of the amenity lists |

  When a feature is missing on either side, its weight is spread over the others.
- **Property level, then floor plan** (3.4): select comps per building, then inside the comp set
  match floor plans by bedrooms, bathrooms and a square-footage band, and price at floor-plan level.
- **Baseline rent** (4.1): similarity-weighted mean of the comps' rent per sq ft × the subject's sq
  ft. With no comp evidence for a unit type, output nothing rather than a default.
- **Evaluation** (3.5 and the usefulness section): precision and recall of our comps against
  analyst-selected sets, which Accolade will provide for a sample of Chicago buildings.
- A **learned embedding** is welcome, but only if it beats the weighted baseline (3.1).

## 2. What we have in the real Dwellsy data

The building is `community_id`. Chicago has 41,953 communities, and half of them are a single unit
(houses, small buildings).

| Comp attribute | Real column or stand-in | Notes |
|---|---|---|
| Location | `latitude`, `longitude` | 100% filled |
| Size | `top_down_community_count` (units in the community) | 7% of rows are single units |
| Vintage | `year_built` | 32% missing; means "built **or** renovated" |
| Amenities | `amenities` tags (250 in Chicago) | Tags vary between agents posting the same unit: use tags seen on most of a community's listings |
| Unit mix | share of listings by bedroom count, per community | From past listings only |
| Type | `uru_type` (Apartment, House, Townhouse, Duplex, Mobile) | Gives the single-family vs multifamily filter |
| Price position | median rent per sq ft over the last months | Useful to display; risky as a selection input (see 4.4) |
| Building form | stand-in: bands of `top_down_address1_count` (units at the street address) | Rough; there is no stories column |
| Lease-up | stand-in: community first seen in the last ~12 months **and** many units listed at once | Needs checking on a few known new buildings |
| Rent control | stand-in for DC: in the District (not MD or VA) and built before 1976 | Check the rule; Illinois law currently bars local rent control, so in Chicago this filter does nothing |
| Interior quality | none | Weight spread over the others, as the plan says (5.2) |
| Class, product type | none | Report as a gap; ask Accolade |

## 3. The baseline pipeline

### Step 1. A building table as of a date

One row per building **per month**, built only from listings created before that month. If the
table uses all listings, a 2022 comp set "knows" amenities and rents from 2025: the same
look-ahead problem we flagged in the building price-history averages.

Columns: location, size, vintage, type, amenity set, unit mix, median rent per sq ft over the last
6 months, number of listings (the evidence count).

### Step 2. Candidates within 5 miles

A haversine BallTree (as in `libs/geo.py`) finds every building within 5 miles of the subject.
That is the plan's location scale.

### Step 3. Hard filters

| Filter | Real column | Default rule to propose |
|---|---|---|
| Single-family vs multifamily | `uru_type`, `top_down_community_count` | Never mix houses and apartment buildings |
| Product type | — (gap) | Same type only, once a source exists |
| Lease-up | stand-in (section 2) | Same status only |
| Class tier | — (gap) | Same tier, or ±1, once a source exists |
| Building form | stand-in (section 2) | Same band |
| Rent control | DC stand-in | Same status only |
| Size | `top_down_community_count` | Optional: between 0.5× and 2× |
| Year built | `year_built` | Optional: within 30 years |

Rule from the plan: a missing value excludes a candidate only when that filter is switched on.

### Step 4. Similarity score

For subject *s* and candidate *c*, with feature scores *fᵢ* and plan weights *wᵢ*, using only the
features present on both sides:

```
similarity(s, c) = Σ wᵢ · fᵢ(s, c) / Σ wᵢ
```

Quality is always missing, so its 0.25 is in practice spread over the other four. Keep the four
feature scores in the output: they are the per-feature breakdown the plan requires, and the
"reasons" for each comp.

### Step 5. Top 10 and floor-plan matching

- Keep the top 10 by similarity.
- Inside those buildings, keep the floor plans with the same bedrooms and bathrooms and a sq ft
  band around the subject's. The band width is still an open question for Accolade.
- Record the evidence count (how many matched floor plans or listings). With none, output nothing.

### Step 6. Comp baseline rent

```
baseline_rent = Σ similarity · comp_rent_per_sqft / Σ similarity  ×  subject_sqft
```

Use only comp rents from before the prediction month. This number is the minimum bar for every
model (plan 4.1), and it doubles as the main test of comp quality (section 5).

## 4. Ideas to go beyond the baseline

Ranked from cheapest to most ambitious. Each must beat the baseline on the rent test in section 5.

### 4.1 Tune the weights (cheap, likely to help)

Keep the plan's formula and search the weights. Try every combination on a 0.05 grid (location,
size, vintage, amenities, summing to 1), and keep the one whose comp baseline has the lowest median
% error on training buildings. Then report the error on held-out buildings. This stays fully
interpretable: "location 0.50, amenities 0.25, ..." is easy to explain to Accolade.

Risk: it optimizes "comps that predict rent", which is close to, but not the same as, "comps an
analyst would pick".

### 4.2 Adjust each comp's rent before averaging (what appraisers do)

A comp that is 200 sq ft bigger, or has a pool, should not count at face value. Adjust each
comp's rent by the difference between it and the subject, using the coefficients or SHAP values of
our rent model ("comp has a pool: −$40"), then average.

This fills two plan requirements at once: a computed default for the per-comp price adjustment
(3.1) and plain-language reasons (4.1).

### 4.3 Let the variable scores choose the attributes

The variable scoring (Gini, gradient boosting, Lasso) shows which attributes actually move rent.
Use it to decide what enters the similarity and with what weight: for example amenity count or
specific amenity tags, unit mix and distance to downtown, on top of the plan's four. In practice
this is a k-nearest-neighbors search in a standardized attribute space, with weights proportional
to importance.

One caution: the scores are per listing and comps are per building, so aggregate the attributes
to the building first.

### 4.4 A learned similarity from pairs

Build training pairs (subject, candidate) whose features are the differences in each attribute
plus the distance. The target is how far apart their rent per sq ft is in later months. A gradient
boosting model then learns which differences matter, and similarity = minus the predicted gap.
This is the "learned" option the plan hopes for, and it can still give a breakdown through SHAP.

Rent per sq ft should be the **target** here, never an **input** to selection. If comps are picked
because their rent is close to the subject's, the comp baseline looks great for a circular reason.

### 4.5 Spatial market segments

Straight-line miles ignore the river, highways, the lake, and the District line in DC. Following
the spatial market segments paper you reviewed, cluster the city into contiguous submarkets (from
ZIPs or census tracts, using housing stock and rent level). Then use "same segment" as a filter or
a bonus. The segments can also be the "custom market boundary" the plan mentions. Free census
data (income, density) would add the "demographics of the surrounding market" from plan 3.2.

### 4.6 Building embeddings (most ambitious)

Represent each building as a vector (amenity tags compressed with SVD, unit mix, size, vintage,
location) and use cosine similarity, or text embeddings of the amenity lists. It is fast with a
nearest-neighbor index, but the hardest to explain. Keep the plan's per-feature breakdown next to
it.

**Suggested order:** baseline → 4.1 → 4.2 or 4.3 → 4.4 to 4.6 only if there is time.

## 5. How to evaluate

| Test | What it tells us |
|---|---|
| Hand-checked examples: filters, feature scores and similarity worked out by hand for 3 to 5 Chicago buildings, saved as unit tests | Our code does what the plan's formula says |
| Comp baseline median % error on held-out buildings and later months | **Comps that predict rent: the main way to compare methods** |
| Coverage: share of unit types with comp evidence | How often we must output nothing |
| Stability: overlap of a building's comp set from one month to the next | Comp sets should not jump around |
| Sanity plots: distance, size ratio and vintage gap of chosen comps; maps for 5 buildings | Catch nonsense; gives the worked example for the presentation (plan 4.3) |
| Recall and precision at 10 vs analyst sets | What Accolade will judge, once they send the sets |

Until the analyst sets arrive, the rent test is the only score that can choose between methods.

## 6. Pitfalls in the real data

- **Single-unit communities.** Half the communities are houses or small buildings. Either give
  them comps only among other single units, or leave them out of v1 and say so.
- **Duplicate postings.** The same unit is often posted by several agents (32% overlap in Chicago).
  Count units, not listings, when computing a community's unit mix, rent and evidence count, ideally
  after vacancy episodes exist.
- **Look-ahead.** Build every attribute and rent from earlier listings only (step 1).
- **Year built is "built or renovated".** A 1920 building renovated in 2015 shows 2015. Keep the
  vintage weight low, or ask Accolade whether the two can be told apart.
- **DC crosses three jurisdictions.** Only 31% of DC-market listings are in the District. A comp
  across a state line may follow different rules and taxes, so consider a jurisdiction filter.
- **Edges of the data.** The Chicago metro includes Indiana and Wisconsin, where comps within 5 miles
  can be missing from the export.

## 7. A possible plan for the next three weeks

| When | What | Who |
|---|---|---|
| By Nov 21 | `libs/comps.py` (point-in-time community table, filters, score, top 10, floor-plan matching) on real Chicago, with unit tests on the hand-checked buildings | Claude writes the spec, Codex implements, Claude reviews |
| By Nov 28 | Comp baseline rent and the rent test on held-out Chicago buildings and months; weight tuning (4.1) | Same split |
| By Dec 5 | One improvement (4.2 or 4.3) if it beats the baseline; DC; worked example and maps for the presentation | Team decides; Codex builds |
| Now | Ask Accolade for analyst comp sets for 20 to 30 Chicago buildings, including our worked example | Team |

Under `CLAUDE.md`, the code parts go through a plan, your OK, `PLAN.md` and Codex.

## 8. Decisions to make

**For the team:**
- Keep the 5-mile radius everywhere, or adapt it to density (dense downtown, sparse suburbs)?
- Which hard filters are on by default, and how wide are the size and year bands?
- Do single-unit communities get comps in v1?
- Is rent per sq ft allowed as a selection input? Our proposal: no, only as display and as the
  evaluation target.

**For Accolade:**
- When can we get the analyst comp sets, and for which buildings?
- How wide is a unit type's sq ft band?
- Is there any source for class, product type, lease-up or rent control in the full feed?
- Should comps cross state lines in DC?
