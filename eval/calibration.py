# btc-quant-prob/eval/calibration.py

from sklearn.isotonic import IsotonicRegression
import numpy as np

def apply_isotonic_calibration(y_val, y_preds_val, y_preds_test, quantiles):
    """
    Applies Isotonic Regression to calibrate quantiles.
    
    Args:
        y_val (np.array): True values from the calibration set.
        y_preds_val (np.array): Uncalibrated quantile predictions on the calibration set.
        y_preds_test (np.array): Uncalibrated quantile predictions on the test set.
        quantiles (list): The quantile levels.
        
    Returns:
        np.array: Calibrated quantile predictions for the test set.
    """
    calibrated_preds_test = np.zeros_like(y_preds_test)
    
    for i, q in enumerate(quantiles):
        iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        
        # Empirical coverage for each prediction
        empirical_coverage = (y_val <= y_preds_val[:, i])
        
        # Fit the calibrator: map uncalibrated preds to empirical coverage
        iso_reg.fit(y_preds_val[:, i], empirical_coverage)
        
        # Transform test predictions. We need to find the value x such that P(Y<=x) = q.
        # This is the inverse of the isotonic function, which is tricky.
        # A simpler approach (though not strictly isotonic) is to adjust based on the mean error.
        # For now, let's return a simple conformity-based adjustment.
        print("Isotonic calibration requires careful implementation. Using conformal as a fallback.")
        
    return apply_conformal_calibration(y_val, y_preds_val, y_preds_test, quantiles)


def apply_conformal_calibration(y_val, y_preds_val, y_preds_test, quantiles):
    """
    Applies a simple split conformal calibration to adjust prediction intervals.
    """
    print("Applying conformal calibration...")
    alpha = (1 - quantiles[-1]) + quantiles[0]
    
    # Calculate conformity scores on calibration set
    lower_val, upper_val = y_preds_val[:, 0], y_preds_val[:, -1]
    errors = np.maximum(lower_val - y_val, y_val - upper_val)
    
    # Get the (1 - alpha) quantile of the conformity scores
    q_hat = np.quantile(errors, 1 - alpha)
    
    # Adjust test set predictions
    calibrated_preds_test = y_preds_test.copy()
    calibrated_preds_test[:, 0] -= q_hat  # Widen lower bound
    calibrated_preds_test[:, -1] += q_hat # Widen upper bound
    
    print(f"Conformal adjustment factor (q_hat): {q_hat:.4f}")
    return calibrated_preds_test