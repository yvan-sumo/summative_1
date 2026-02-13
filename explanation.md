Cell 0 — Title & Overview (markdown)
Just the introduction text explaining what the project is about and why regularisation makes sense for this dataset.

Cell 2 — Imports
Like unpacking your toolbox before starting work. You're loading all the libraries you'll need — numpy for maths, pandas for tables, matplotlib/seaborn for charts, and sklearn for the actual machine learning models.

Cell 3 — Load the data
Downloads the red wine and white wine CSV files from the internet, sticks them together into one big table, and adds a column called is_red (1 = red, 0 = white) so the model knows which type each row is. You end up with 6,497 rows.

Cell 5 — Basic checks
Quickly checks: are there any missing values? (No.) And how many wines got each quality score? Most wines scored 5 or 6 — very few scored 3 or 9.

Cell 6 — Two charts
Draws two plots side by side: a bar chart of quality score counts, and a heatmap showing how correlated all the features are with each other. The heatmap is the key one — it visually proves multicollinearity exists (e.g. alcohol and density move together).

Cell 8 — Feature engineering
This is where you "expand" the features. You start with 12 columns and create new ones by squaring each feature and multiplying pairs together (e.g. alcohol × density). You go from 12 features to 90. More correlated features = more reason to use regularisation.

Cell 9 — Train/test split + scaling
Splits the data: 80% for training the models, 20% kept aside for testing. Then scales all features to have mean=0 and the same spread — this is essential for regularisation to work fairly, otherwise features measured in big numbers would get unfairly penalised more.

Cell 11 — OLS (baseline model)
Fits a plain old linear regression with no penalty. This is the baseline — the "do nothing special" model you compare everything else against. It uses all 90 features with no restrictions.

Cell 12 — Ridge
Fits Ridge regression. It tries 300 different values of alpha (the penalty strength) using cross-validation to find the best one, then trains the final model with that alpha. Ridge shrinks all coefficients toward zero but never removes any feature completely.

Cell 13 — Lasso
Same idea as Ridge but with a different penalty. Lasso is more aggressive — it can shrink some coefficients all the way to exactly zero, effectively removing those features. In your results it kept only 28 out of 90 features.

Cell 14 — Elastic Net
A blend of Ridge and Lasso. It tunes two things via CV: alpha (penalty strength) and l1_ratio (how much Lasso vs Ridge to mix in). In your case it picked l1_ratio=1.0, meaning it decided pure Lasso was best.

Cell 16 — Results table
Builds a neat table comparing all four models by RMSE and R² on the test set.

Cell 17 — Bar chart + predicted vs observed
Two plots: a bar chart comparing RMSE across models, and a scatter plot of predicted quality vs actual quality for the best model. Ideally points follow the diagonal line.

Cell 19 — Ridge coefficient path
Trains Ridge 120 times with different alpha values and plots how each coefficient changes. As alpha increases, all lines shrink smoothly toward zero. The red dashed line shows where the CV-chosen alpha landed.

Cell 20 — Lasso coefficient path + sparsity plot
Same idea but for Lasso. Left plot: coefficients get zeroed out as alpha increases. Right plot: shows how many features survive as alpha gets stronger — at your optimal alpha, only 28 remain.

Cell 21 — Top Lasso features
Bar chart showing the 28 features Lasso kept, ranked by importance. Green = positive effect on quality, red = negative effect.

Cell 23 — CV error curve
Shows the U-shaped bias-variance trade-off curve for Lasso. Left side = model overfits (too flexible), right side = model underfits (too restricted), the bottom of the U = sweet spot where your optimal alpha sits.

Cell 25 — Residual diagnostics
Three plots checking whether the best model's errors look "well-behaved": residuals vs fitted values (should be random scatter around zero), a histogram of errors (should look roughly bell-shaped), and a Q-Q plot (checks if errors are normally distributed).

Cell 26 — Discussion (markdown)
The written reflection — this is the cell that needs updating based on your actual results (as we discussed).

Cell 27 — Final summary printout
Prints a clean table summarising all model results including the optimal alpha values found by CV.