"""Example of treehfd for multiclassification."""

# Load packages.
import logging

import numpy as np
import xgboost as xgb
from numpy.random import default_rng

from treehfd import XGBTreeHFD

# Set up logger.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # Generate simulated data.
    DIM = 6
    NSAMPLE = 5000
    RHO = 0.5
    mu = np.zeros(DIM)
    cov = np.full((DIM, DIM), RHO)
    np.fill_diagonal(cov, np.ones(DIM))
    X = default_rng().multivariate_normal(mean=mu, cov=cov, size=NSAMPLE)
    y = np.sin(2*np.pi*X[:, 0]) + X[:, 0]*X[:, 1] + X[:, 2]*X[:, 3]
    y += default_rng().normal(loc=0.0, scale=0.5, size=NSAMPLE)
    label = np.zeros(NSAMPLE)
    label[y > 0] = 1
    label[y > 1] = 2

    # Fit XGBoost model.
    xgb_model = xgb.XGBClassifier(eta=0.05, n_estimators=300, max_depth=3,
                                  objective="multi:softmax")
    xgb_model = xgb_model.fit(X, label)

    # Generate testing data.
    X_new = default_rng().multivariate_normal(mean=mu, cov=cov, size=NSAMPLE)
    y_new = (np.sin(2*np.pi*X_new[:, 0]) + X_new[:, 0]*X_new[:, 1]
            + X_new[:, 2]*X_new[:, 3])
    y_new += default_rng().normal(loc=0.0, scale=0.5, size=NSAMPLE)
    label_new = np.zeros(NSAMPLE)
    label_new[y_new > 0] = 1
    label_new[y_new > 1] = 2

    # Accuracy of XGBoost.
    xgb_labels = xgb_model.predict(X_new)
    xgb_acc = np.sum(label_new == xgb_labels) / NSAMPLE
    msg_acc = "Classification accuracy of XGBoost model: {xgb_acc}"
    logger.info(msg_acc)

    # Fit TreeHFD.
    treehfd_model = XGBTreeHFD(xgb_model)
    treehfd_model.fit(X, interaction_order=2)

    # Compute TreeHFD predictions.
    xgb_pred = xgb_model.predict(X_new, output_margin=True)
    xgb_labels = xgb_model.predict(X_new)
    xgb_acc = np.sum(label_new == xgb_labels) / NSAMPLE
    y_main, y_order2 = treehfd_model.predict(X_new)
    hfd_logits, mse_resid = np.zeros((NSAMPLE, 3)), np.zeros(3)
    for label in range(3):
        hfd_pred = (treehfd_model.eta0[label] + np.sum(y_main[:, label, :],
                    axis=1) + np.sum(y_order2[:, label, :], axis=1))
        hfd_logits[:, label] = hfd_pred
        resid = xgb_pred[:, label] - hfd_pred
        mse_resid[label] = np.round(np.mean(resid**2) / np.var(
                               xgb_pred[:, label]), decimals=3)
    hfd_labels = np.argmax(hfd_logits, axis=1)
    hfd_fid = np.sum(hfd_labels == xgb_labels) / NSAMPLE
    hfd_acc = np.sum(hfd_labels == label_new) / NSAMPLE

    # Print performance results.
    msg_resid = f"Normalized MSE of TreeHFD residuals: {mse_resid}"
    logger.info(msg_resid)
    msg_acc_xgb = f"Accuracy of XGBoost model: {xgb_acc}"
    logger.info(msg_acc_xgb)
    msg_fidelity = f"Fidelity of TreeHFD to XGBoost model: {hfd_fid}"
    logger.info(msg_fidelity)
    msg_acc_hfd = f"Accuracy of TreeHFD model: {hfd_acc}"
    logger.info(msg_acc_hfd)
