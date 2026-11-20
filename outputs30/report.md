# Classification report: dx from 30 cell-nucleus features

## Data overview
- 568 rows and 30 numeric features, with no missing values, so no imputation was needed.
- Target dx: B 357 and M 211 (about 63% benign, 37% malignant). The majority-class baseline accuracy is 0.6316.
- Strongly right-skewed features: area_se (skew 5.518, max 542.2 vs median 24.485), concavity_se (skew 5.116), fractal_dim_se (skew 3.933), perimeter_se (skew 3.489) and radius_se (skew 3.123). Log or robust scaling is worth considering for these.
- Six concavity-type features have a minimum of 0 (concavity, concave_points, concavity_se, concave_points_se, concavity_wt, concave_points_wt), which may be a measurement floor.
- Figures: outputs/figures/feature_distributions.png and outputs/figures/features_by_target.png.

## Correlated features
- The |r| >= 0.8 filter flags 42 pairs. The size group is almost entirely redundant: radius vs perimeter r = 0.9979, radius_wt vs perimeter_wt r = 0.9938, radius vs area r = 0.9873, perimeter vs area r = 0.9865 and radius_wt vs area_wt r = 0.9839.
- Concavity, concave_points and compactness are also strongly intercorrelated (concavity vs concave_points r = 0.9205, compactness vs concavity r = 0.8816).
- The greedy filter (threshold 0.8, computed on training rows) dropped 17 of 30 features and kept 13. Its pair correlations differ slightly from the full-data matrix because they were computed on training rows only.
- Figure: outputs/figures/correlation_heatmap.png.

## Model performance
- **all_features (30 features):** CV accuracy 0.9736 ± 0.0241 and CV ROC-AUC 0.9909 ± 0.0158. On validation, accuracy is 0.9825, balanced accuracy 0.9812 and ROC-AUC 0.9974. The confusion matrix is [[71,1],[1,41]], so there are 2 errors.
- **uncorrelated_features (13 features):** CV accuracy 0.9604 ± 0.0264 and CV ROC-AUC 0.9885 ± 0.0165. On validation, accuracy is 0.9825, balanced accuracy 0.9762, ROC-AUC 0.9927 and macro F1 0.981. The confusion matrix is [[72,0],[2,40]], so there are 2 errors.
- Both runs beat the 0.6316 baseline by about 35 percentage points.
- CV and validation agree that performance is high. Validation accuracy (0.9825) sits at or above both CV means. Fold accuracies ranged from 0.9341 to 0.989 for the full run and from 0.9231 to 0.9889 for the reduced run.
- The validation set has only 114 rows, so these estimates are noisy.

## Full vs. uncorrelated features
- Reduced minus full: CV accuracy -0.0132, CV ROC-AUC -0.0024, validation accuracy 0.0 and validation ROC-AUC -0.0047.
- Per-fold accuracy differences (reduced minus full) are +0.011, -0.011, -0.0439, -0.022 and 0.0. The signs are mixed: the reduced run wins one fold, loses three and ties one. The mean gap is smaller than the fold-to-fold standard deviations (0.024 and 0.026), so the difference is within fold-to-fold noise rather than a consistent effect.
- Dropped features (17): concavity, concave_points, compactness, concave_points_wt, concavity_wt, perimeter_wt, radius_wt, compactness_se, perimeter_se, compactness_wt, area_wt, radius_se, smoothness, perimeter, radius, texture_wt and area_se. Each was removed as a member of a highly correlated pair, choosing the member with the highest mean |r| to the other features.
- SHAP shares shifted once the correlated features were removed. Importance concentrated in the remaining features: area rose from a share of 0.026 to 0.3239 and became the top feature. Other increases were smoothness_wt (0.0428 to 0.1185), concave_points_se (0.0258 to 0.1116), texture (0.0289 to 0.0913), symmetry_wt (0.0474 to 0.0802), fractal_dim_wt (0.0371 to 0.0712) and fractal_dim_se (0.0312 to 0.0682).
- **Recommendation: all_features.** It has the higher CV accuracy (0.9736 vs 0.9604), higher CV ROC-AUC (0.9909 vs 0.9885), higher validation ROC-AUC (0.9974 vs 0.9927) and higher balanced accuracy (0.9812 vs 0.9762). Validation accuracy is identical. The reduced run is more parsimonious (13 vs 30 features) and its SHAP story is simpler, and it is a reasonable choice if interpretability matters more than the small performance gap. Its malignant-class misses (2 false negatives vs 1 for the full run) favour the full run.
- Figures: outputs/figures/all_features_shap_importance.png, outputs/figures/uncorrelated_features_shap_importance.png.

## What drives the predictions
The recommended run is all_features. SHAP values are log-odds contributions toward M. The top features by mean absolute SHAP are:
- texture_wt 1.0462 (share 0.0787), positive: higher values push toward M.
- radius_wt 0.7936 (share 0.0597), positive.
- concave_points 0.77 (share 0.0579), positive.
- radius_se 0.7468 (share 0.0562), positive.
- area_wt 0.7129 (share 0.0536), positive.
- Next come perimeter_wt (0.664), symmetry_wt (0.6296), concavity_wt (0.624) and concave_points_wt (0.6027), all positive.
- compactness_se 0.6189 (share 0.0466) is the leading negative feature: higher values push toward B. fractal_dim_se (0.415) is also negative.

The top five features together account for about 0.31 of total importance, so no single feature dominates. The coefficients agree with SHAP: the largest positive coefficients are radius_se 1.2013, texture_wt 1.1829 and concave_points 0.9936, and the negative ones are compactness_se -0.9064 and fractal_dim_se -0.6643.
- Figure: outputs/figures/all_features_shap_beeswarm.png.

## Example predictions
- **Row 179** (confident correct, true M, predicted M with probability 1.0): very large size and concavity values drive the prediction. area_wt = 3216 (99.3rd percentile) adds +3.791, concave_points = 0.1878 adds +3.538, radius_wt = 33.12 adds +3.378, perimeter_wt = 220.8 adds +2.774 and area = 2250 adds +2.029.
- **Row 454** (borderline, true B, predicted M with probability 0.5065): texture_wt = 41.61 (98.7th percentile, +3.101) and texture = 30.72 (+1.240) push toward M. symmetry_wt = 0.2196 (-0.867) and concavity_wt = 0.07003 (-0.793) push back toward B, leaving the probability near 0.5.
- **Row 39** (confident misclassification, true M, predicted B with probability 0.934): smoothness_wt = 0.1094 (-0.748), concave_points = 0.02031 (25.8th percentile, -0.723) and radius_se = 0.2385 (-0.709) push toward B. texture_wt = 30.25 (+0.872) and compactness_se = 0.01102 (+0.725) push toward M, but not enough to overcome them.
- **Row 196** (correct, near the boundary, true M, predicted M with probability 0.6998): smoothness_wt = 0.08822 (1.3rd percentile, -1.450) and compactness_se = 0.05296 (92.1st percentile, -1.346) push toward B. radius_se = 0.6362 (+0.950) and radius_wt = 19.76 (+0.707) push toward M.

## Caveats
- SHAP splits credit arbitrarily among correlated features, such as radius, perimeter and area, or texture and texture_wt. Individual shares in the full run understate how much the size group matters collectively.
- A linear logistic model cannot capture interactions or threshold effects between features.
- The validation set is small: 114 rows, and each single error moves validation accuracy by under one percentage point. Each run has only 2 validation errors, so differences between runs are not statistically meaningful.
- SHAP magnitudes are not comparable across runs. The reduced run's area contribution for row 179 is 17.5293 log-odds, so compare shares rather than raw values.
- Heavily skewed _se features and zero-valued concavity features were not transformed, which may affect the L2-regularized coefficients (C = 1.0).
- Feature selection used correlation only. Dropped features may carry target-relevant information, as the reasons' target-association values show.
