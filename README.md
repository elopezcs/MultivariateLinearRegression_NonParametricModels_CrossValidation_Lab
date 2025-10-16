# Multivariate Linear Regression, Non-Parametric Models, and Cross-Validation

Predict diabetes progression using Scikit-Learn. Compare multivariate Polynomial Regression, Decision Trees, and KNN. Use a fixed train, validation, and test split. Report R², MAE, and MAPE. Include a bonus Logistic Regression classifier for a high-risk label.

## Notebook
`MultivariateLinearRegression_Lab.ipynb`

## Objective
Build a screening model for diabetes progression one year after baseline. Prepare data splits and evaluate several models. Use the best metrics for model selection and discussion.

## Dataset
Scikit-Learn Diabetes dataset. 442 samples and 10 standardized features. Target is a continuous score named `disease_progression`.

## Methods
- **Data split**. 75% train, 10% validation, 15% test, with fixed seeds.
- **Models**.
  - Polynomial Regression, degrees 2 and 3.
  - Decision Tree Regressors, depths 3 and 5, plus plots and importances.
  - KNN Regressors, k = 3 and 7, with scaling.
  - Logistic Regression, binary high-risk label from top third of train target. Metrics: Accuracy, Precision, Recall, ROC-AUC.
- **Feature selection**. Select features above the median importance from the best validation tree, then retrain KNN, Tree, and Poly d=2 on the reduced set.

## Key Results
**Regression, test set**  
The table below is produced in the notebook.

| Model | R² | MAE | MAPE |
|---|---:|---:|---:|
| KNN k=7 **(selected)** | **0.535** | 41.804 | 35.593 |
| Poly d=2 **(selected)** | 0.519 | 42.231 | 35.485 |
| KNN k=7 | 0.479 | 41.478 | 35.164 |
| Tree d=3 | 0.438 | 46.115 | 40.176 |
| Poly d=2 | 0.399 | 45.493 | 37.949 |
| Tree d=5 | 0.366 | 47.737 | 39.530 |
| Poly d=3 | −65.832 | 243.156 | 213.852 |

**Takeaways**  
KNN k=7 gives the best R² after feature selection. Polynomial d=2 is close and easier to explain. Deeper trees and degree 3 overfit or add complexity without gains.

**Selected features**  
`bmi, bp, s2, s4, s5`.

**Binary high-risk classifier**  
Accuracy 0.806, Precision 0.737, Recall 0.636, ROC-AUC 0.861 on the test set. The label is top one-third of train target.

## What You Will Learn
- How to structure a 75-10-15 split and keep seeds fixed.
- How to compare R², MAE, and MAPE across models.
- How to use tree importances for simple feature selection.
- How to add a simple clinical threshold for a binary risk label.

## Environment
- Python 3.12.x  
- Packages used in the notebook include: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `ipython` display utilities.

### Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repo.  
2. Create and activate a virtual environment.  
3. Install dependencies. 
4. Launch Jupyter and open `MultivariateLinearRegression_Lab.ipynb`.  
5. Run all cells from top to bottom.

## Repository Structure
```
.
├── MultivariateLinearRegression_Lab.ipynb
└── README.md
```

## Reproduce the Results
- Keep the split proportions and `random_state=42` to match the reported tables.
- Run the multivariate sections to generate validation and test tables for Poly, Trees, and KNN.
- Run the feature selection cell to print the selected feature list and retrained metrics.
- Run the Logistic Regression cell to print the classification metrics.

## Notes and Tips
- Monitor degree 3 polynomial for overfitting on small data. The notebook shows a large negative R². Prefer degree 2.
- Scale inputs for KNN. The notebook uses a pipeline with `StandardScaler`.
- Use feature selection to reduce variance and keep interpretation simple. Verify that R² and error metrics do not degrade.

