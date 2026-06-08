# Error Sample Analysis

This folder keeps the final analysis reports and useful support files.

## Final notebooks

- `model_error_and_metrics_analysis.ipynb`: main V1 metric analysis and typical error sample analysis. This is the primary report and follows the original image-plus-interpretation format.
- `v1_v2_comparison_analysis.ipynb`: V1/V2 comparison report. It follows the same visual style, with figures, interpretation under each figure, and the V2 improvement plan.

## Support files worth keeping

- `figures/`: generated comparison figures used by `v1_v2_comparison_analysis.ipynb`.
- `supporting_files/optimization_plan.md`: V2 optimization plan and collaboration notes.
- `supporting_files/supplement_candidates_report.md`: supplement data source and count report.
- `supporting_files/corresponding_data_report.md`: corresponding-data lookup report.
- `supporting_files/tables/overall_metrics.csv`: V1 overall metrics.
- `supporting_files/tables/per_class_metrics.csv`: V1 per-class metrics.
- `supporting_files/tables/top_error_pairs.csv`: V1 major error pairs.
- `supporting_files/tables/high_confidence_error_samples.csv`: V1 high-confidence error examples.
- `supporting_files/tables/hard_examples_for_v2.csv`: optional input for rerunning V2 with hard-example weighting.
- `supporting_files/tables/supplement_candidates_manifest.csv`: supplement sample manifest.
- `valuable_extra_files_review.csv`: review list explaining which extra files are worth keeping or can be deleted.
