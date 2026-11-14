# Data source and licence

## Approved input for this release

- **Dataset:** [Deutsche Bahn Data on Hugging Face](https://huggingface.co/datasets/piebro/deutsche-bahn-data) — `monthly_processed_data/data-YYYY-MM.parquet`.
- **Historical collection and processing:** Piet Brömmel ([source and documentation](https://github.com/piebro/deutsche-bahn-data)).
- **Underlying data:** Deutsche Bahn Station Data / Timetables APIs.
- **Published data licence:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Modification:** station filtering to Dortmund; select published fields; derive planned-vs-changed arrival differences; aggregate.
- **Redistribution:** no source or processed row-level Parquet in this GitHub repository.

## Attribution to carry into any public derived work

> Underlying timetable-update data © Deutsche Bahn, used under CC BY 4.0. Historical collection and processing by Piet Brömmel, [Deutsche Bahn Data](https://github.com/piebro/deutsche-bahn-data), [Hugging Face archive](https://huggingface.co/datasets/piebro/deutsche-bahn-data). Data filtered to Dortmund railway stations and transformed into aggregated analytical summaries. Independent, non-endorsed educational analysis.

Source metadata and downloaded release versions should be recorded when real data is processed. Licences and provider usage conditions should be rechecked before expanding or redistributing data.

## Deliberately excluded

VRR feeds, other third-party archives, passenger and employee records, weather, location and traffic feeds. Adding sources requires separate review.

This documentation explains the intended licence compliance approach, not legal advice or a guarantee against all disputes.
