# Classification report: diagnosis (dx) from 10 cell-nucleus features

## 1. Data overview
- 568 rows, 10 numeric features, no missing values. Target dx has 357 B (benign) and 211 M (malignant), about 63% benign, a moderate imbalance.
- Right skew is pronounced for area (skew 1.652), concavity (1.407), fractal_dim (1.313), concave_points (1.175) and compactness (1.177). Smoothness (0.460) and symmetry (0.727) are closer to symmetric.
- Concavity has a minimum of 0.0, so some samples have no measured concavity.
- Figures: outputs/figures/feature_distributions.png and outputs/figures/features_by_target.png.

## 2. Correlated features
The threshold was |r| >= 0.8 (Pearson). Nine pairs exceed it:
- Size family: radius vs perimeter 0.9979, radius vs area 0.9873, perimeter vs area 0.9865.
- Concavity family: concavity vs concave_points 0.9205, compactness vs concavity 0.8816, perimeter vs concave_points 0.8513, compactness vs concave_points 0.8291, area vs concave_points 0.8242, radius vs concave_points 0.8232.
- The feature-selection table reports slightly different values for some pairs (radius vs area 0.9869, concave_points vs concavity 0.9197, concavity vs compactness 0.8847). These differ from the matrix by under 0.003 and do not change any decision.
- See outputs/figures/correlation_heatmap.png.

## 3. Model performance
Majority-class baseline accuracy is 0.6316.

| Run | Features | CV accuracy (mean ± std) | CV ROC-AUC (mean ± std) | Validation accuracy | Validation ROC-AUC |
|---|---|---|---|---|---|
| all_features | 10 | 0.9383 ± 0.0126 | 0.9857 ± 0.0083 | 0.9386 | 0.9858 |
| uncorrelated_features | 6 | 0.9361 ± 0.0197 | 0.9805 ± 0.0090 | 0.9298 | 0.9818 |

- Validation balanced accuracy is 0.9415 for all_features and 0.9246 for uncorrelated_features.
- Both runs beat the baseline by about 30 points. Validation accuracy sits at or within the CV spread for both runs, so CV and validation agree.
- Confusion matrices (rows B, M; columns predicted B, M): all_features [[67,5],[2,40]]; uncorrelated_features [[68,4],[4,38]].

## 4. Full vs. uncorrelated features
- Differences (uncorrelated minus full): CV accuracy -0.0022, CV AUC -0.0052, validation accuracy -0.0088, validation AUC -0.0040.
- Per-fold accuracy differences are -0.022, +0.0219, -0.011, 0.0 and 0.0. The signs are mixed and the magnitudes are within fold-to-fold noise, so removing the correlated features does not clearly hurt or help accuracy.
- Dropped features: concave_points, concavity, perimeter and radius. Each was removed for high correlation with a retained feature and a high mean |r| to the others.
- SHAP importance shifted. In the full run, concave_points (0.1946), radius (0.1105), perimeter (0.1008) and concavity (0.0868) shared the credit that is now concentrated elsewhere. Area's share rose from 0.1215 to 0.414, and compactness's share rose from 0.0218 to 0.1515. Texture (0.1652 to 0.1677), smoothness (0.1315 to 0.1469), fractal_dim (0.0429 to 0.0779) and symmetry (0.0245 to 0.042) changed little.
- Recommendation: all_features. It has higher CV accuracy and AUC and higher validation accuracy and AUC, and its CV standard deviation is lower (0.0126 vs 0.0197). The uncorrelated run is more parsimonious (6 features) and its attributions are easier to read because credit is not split among collinear features. If interpretability matters more than the small accuracy gain, uncorrelated_features is a defensible alternative.
- Figures: outputs/figures/all_features_shap_importance.png, outputs/figures/uncorrelated_features_shap_importance.png and the matching beeswarm plots.

## 5. What drives the predictions
From the recommended run (all_features), mean absolute SHAP in log-odds for class M:
1. concave_points: 1.4011 (share 0.1946), higher values push toward M.
2. texture: 1.1897 (0.1652), higher values push toward M.
3. smoothness: 0.9466 (0.1315), higher values push toward M.
4. area: 0.8745 (0.1215), higher values push toward M.
5. radius: 0.7955 (0.1105), higher values push toward M.
6. perimeter: 0.7257 (0.1008), higher values push toward M.
7. concavity: 0.6252 (0.0868), higher values push toward M.
8. fractal_dim: 0.3086 (0.0429), higher values push toward B.
9. symmetry: 0.1768 (0.0245), higher values push slightly toward M.
10. compactness: 0.1568 (0.0218), higher values push slightly toward B.

The size and concavity features carry most of the signal, but they are collinear, so their individual shares are partly arbitrary. Figure: outputs/figures/all_features_shap_importance.png.

## 6. Example predictions
- Row 179 (confident correct, true M, predicted M with probability 1.0): concave_points 0.1878 (99.6th percentile, +6.4373), area 2250 (+5.1367), radius 27.22 and perimeter 182.1 (both at the 99.6th percentile). Very large size and concave-point values drove a strong malignant prediction.
- Row 374 (borderline, true B, predicted B with probability 0.5201): texture 16.07 (23.1st percentile, -1.1249) pulled toward B, while radius 16.17, perimeter 106.3 and area 788.5 (all above the 76th percentile) pushed toward M. The prediction is close to a coin flip.
- Row 39 (confident misclassification, true M, predicted B with probability 0.9596): concave_points 0.02031 (25.8th percentile, -1.3148) and smoothness 0.08162 (15.4th percentile, -1.2626) pushed strongly toward B. Texture 21.58 (+0.7408) pushed toward M but was not enough.
- Row 127 (misclassified, true B, predicted M with probability 0.9181): concave_points 0.08534 (81.1st percentile, +1.695) and smoothness 0.115 (90.7th percentile, +1.6629) pushed toward M. Texture 16.39 (-1.0165) partly offset this.

## 7. Caveats
- SHAP splits credit arbitrarily between correlated features, so per-feature shares for the size and concavity families should be read as a group, not as independent effects.
- A linear logistic model cannot capture interactions or non-linear thresholds.
- The validation set has 114 rows, so each error changes validation accuracy by about 0.9 points. Differences of under 0.01 between runs are not meaningful on their own.
- The class balance (about 63% benign) means accuracy alone can flatter the model, so balanced accuracy and AUC are reported alongside it.
- Skewed features were standardized but not log-transformed, which may affect linear fit on the long right tails of area and concavity.
- Coefficients and SHAP values are on the standardized scale and in log-odds units, so they are not directly comparable to raw-unit effects.