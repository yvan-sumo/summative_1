# sem4_topic1_formative1_tests.py
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler


# ----------------------------
# TODO 1: RidgeCV
# ----------------------------
def test_ridgecv_setup(ridge_cv, alphas):
    """Test TODO 1: RidgeCV is configured & fitted."""
    assert isinstance(ridge_cv, RidgeCV), "ridge_cv must be a sklearn RidgeCV instance."
    assert hasattr(ridge_cv, "alphas"), "ridge_cv must have an alphas attribute."
    assert len(ridge_cv.alphas) == len(
        alphas
    ), "ridge_cv.alphas should match the provided alphas grid."

    # Public: ensure fitted (alpha_ exists) and positive
    assert hasattr(ridge_cv, "alpha_"), "ridge_cv must be fitted (missing alpha_)."
    assert np.isfinite(ridge_cv.alpha_), "ridge_cv.alpha_ must be finite."
    assert ridge_cv.alpha_ > 0, "ridge_cv.alpha_ must be > 0."

    


# ----------------------------
# TODO 2: Ridge fit
# ----------------------------
def test_ridge_model_fit(ridge_model, X_scaled, log_Y):
    """Test TODO 2: Ridge model fitted; coef/intercept shapes consistent."""
    assert isinstance(
        ridge_model, Ridge
    ), "ridge_model must be a sklearn Ridge instance."
    assert hasattr(ridge_model, "coef_"), "ridge_model must be fitted (missing coef_)."
    assert hasattr(
        ridge_model, "intercept_"
    ), "ridge_model must be fitted (missing intercept_)."

    p = X_scaled.shape[1]
    coef = np.asarray(ridge_model.coef_).ravel()
    inter = np.asarray(ridge_model.intercept_).ravel()

    assert coef.shape == (p,), f"Expected {p} coefficients, got {coef.shape}."
    assert inter.shape in [(1,), ()], "Intercept should be scalar-like."

    # Public: predictions finite
    preds = ridge_model.predict(X_scaled[:10])
    assert np.all(np.isfinite(preds)), "Ridge predictions should be finite."

    


# ----------------------------
# TODO 3: Bootstrap OLS vs Ridge
# ----------------------------
def test_bootstrap_implementation(boot_ols, boot_ridge, n_boot, X_cols):
    """Test TODO 3: bootstrap arrays filled; ridge more stable than OLS (median SE)."""
    assert isinstance(boot_ols, np.ndarray), "boot_ols must be a numpy array."
    assert isinstance(boot_ridge, np.ndarray), "boot_ridge must be a numpy array."
    assert (
        boot_ols.shape == boot_ridge.shape
    ), "boot_ols and boot_ridge must have the same shape."

    assert boot_ols.shape[0] == n_boot, "boot_ols must have n_boot rows."
    assert (
        boot_ols.shape[1] == len(X_cols) + 1
    ), "boot arrays must store intercept + all coefficients."

    assert np.isfinite(boot_ols).all(), "boot_ols contains NaN or inf."
    assert np.isfinite(boot_ridge).all(), "boot_ridge contains NaN or inf."

    assert np.abs(boot_ols).sum() > 0, "boot_ols looks empty (all zeros)."
    assert np.abs(boot_ridge).sum() > 0, "boot_ridge looks empty (all zeros)."

    # Public: median coefficient SD should be smaller for ridge than OLS (exclude intercept)
    ols_se = boot_ols[:, 1:].std(axis=0)
    ridge_se = boot_ridge[:, 1:].std(axis=0)
    assert (
        np.median(ridge_se) <= np.median(ols_se) + 1e-12
    ), "Expected Ridge bootstrap coefficient variability (median SD) <= OLS."

    


# ----------------------------
# TODO 4: statsmodels OLS design matrices
# ----------------------------
def test_sm_ols_design_and_fit(X_train, X_test, X_train_const, X_test_const, ols_model):
    """Test TODO 4: constant handled correctly and OLS model fitted."""
    assert isinstance(X_train_const, np.ndarray), "X_train_const must be a numpy array."
    assert isinstance(X_test_const, np.ndarray), "X_test_const must be a numpy array."

    assert (
        X_train_const.shape[0] == X_train.shape[0]
    ), "X_train_const row count must match X_train."
    assert (
        X_test_const.shape[0] == X_test.shape[0]
    ), "X_test_const row count must match X_test."

    p = X_train.shape[1]
    assert X_train_const.shape[1] in (
        p,
        p + 1,
    ), "X_train_const should have p or p+1 columns depending on constant detection."
    assert X_test_const.shape[1] in (
        p,
        p + 1,
    ), "X_test_const should have p or p+1 columns depending on constant detection."

    # Public: fitted model and predictions finite
    assert hasattr(
        ols_model, "params"
    ), "ols_model must be a fitted statsmodels OLS result."
    assert (
        len(ols_model.params) == X_train_const.shape[1]
    ), "OLS params length must match design matrix columns."
    pred = ols_model.predict(X_test_const[:5])
    assert np.all(np.isfinite(pred)), "OLS predictions must be finite."

    


# ----------------------------
# TODO 5: Ridge bootstrap prediction intervals
# ----------------------------
def test_ridge_bootstrap_prediction_intervals(
    ridge_preds_boot, n_boot_pred, X_test, ridge_lo, ridge_hi
):
    """Test TODO 5: bootstrap preds have correct shape; intervals ordered and finite."""
    assert isinstance(
        ridge_preds_boot, np.ndarray
    ), "ridge_preds_boot must be a numpy array."
    assert ridge_preds_boot.shape == (
        n_boot_pred,
        len(X_test),
    ), "ridge_preds_boot has wrong shape."
    assert np.isfinite(ridge_preds_boot).all(), "ridge_preds_boot contains NaN or inf."

    assert isinstance(ridge_lo, np.ndarray) and ridge_lo.shape == (
        len(X_test),
    ), "ridge_lo must be shape (len(X_test),)."
    assert isinstance(ridge_hi, np.ndarray) and ridge_hi.shape == (
        len(X_test),
    ), "ridge_hi must be shape (len(X_test),)."
    assert (
        np.isfinite(ridge_lo).all() and np.isfinite(ridge_hi).all()
    ), "Intervals contain NaN or inf."
    assert np.all(ridge_lo <= ridge_hi), "Lower interval bound must be <= upper bound."

    # Public: non-degenerate widths for many points
    width = ridge_hi - ridge_lo
    assert (
        np.percentile(width, 90) > 0
    ), "Prediction intervals seem degenerate (zero width)."

    
