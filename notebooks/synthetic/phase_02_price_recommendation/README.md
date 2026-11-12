# Phase 2: price recommendation

The planned Phase 2 notebooks follow §5 of `docs/project_phases.md`. **No Phase 2 code exists yet.**

| Notebook | Input tables | Answer keys / reference outputs |
|---|---|---|
| `01_concession_extraction` | `market.concession_event` (specials text) | `truth.concession` |
| `02_net_effective_rent` | Asking rent from `market.price_observation`; concession terms from `market.concession_event` | `truth.concession.net_effective_rent`, `truth.lease_event.true_effective_rent` |
| `03_demand_conversion` | `pms.occupancy_target`, `pms.property_settings`, `pms.lease`, `pms.unit`, `pms.guestcard`, `pms.guestcard_event`, `pms.daily_weather`, `pms.weather_alert`, Phase 1 comps and `pms.comp_*` | `truth.demand_params`, `pms.price_recommendation`, `pms.price_recommendation_details` |
| `04_occupancy_forecast` | `pms.lease`, `pms.renewal_offer` | Held-out occupancy outcomes from `pms.lease` |
