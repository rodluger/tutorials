import numpy as np
import matplotlib.pyplot as plt
import celerite
from ipywidgets import (
    interact,
    interactive,
    fixed,
    interact_manual,
    interactive_output,
)
import ipywidgets as widgets
from ipywidgets import Layout
import pandas as pd
from scipy.special import legendre
import celerite
import pandas as pd

try:
    get_ipython().magic('config InlineBackend.figure_format = "retina"')
except:
    pass


def L1(A, y, log_lam=5.0, maxiter=9999, eps=1e-15, tol=1e-8):
    """L1 regularized least squares via iterated ridge (L2) regression.

    See Section 2.5 of

        https://www.cs.ubc.ca/~schmidtm/Documents/2005_Notes_Lasso.pdf

    The basic idea is to iteratively zero out the prior on the weights
    until convergence.

    Args:
        A (ndarray): The design matrix for the linear problem, shape ``(N, M)``.
        y (ndarray): The data vector, length ``N``.
        log_lam (float or ndarray, optional): The log of the regularization
            strength parameter, ``lambda``. This may either be a scalar or
            a vector of length ``N``. Defaults to 5.0.
        maxiter (int, optional): Maximum number of iterations. Defaults to 9999.
        eps (float, optional): Precision of the algorithm. Defaults to 1e-15.
        tol (float, optional): Iteration stop tolerance. Defaults to 1e-8.

    Returns:
        ndarray: The vector of coefficients ``w`` that minimizes the L1 norm
            for the linear problem.
    """
    ATA = A.T @ A
    ATy = A.T @ y
    w = np.ones_like(ATA[0])
    lam = 10 ** log_lam
    for n in range(maxiter):
        absw = np.abs(w)
        if hasattr(lam, "__len__"):
            absw[absw < lam * eps] = lam[absw < lam * eps] * eps
        else:
            absw[absw < lam * eps] = lam * eps
        KInv = np.array(ATA)
        KInv[np.diag_indices_from(KInv)] += lam / absw
        try:
            w_new = np.linalg.solve(KInv, ATy)
        except np.linalg.LinAlgError:
            w_new = np.linalg.lstsq(KInv, ATy, rcond=None)[0]
        chisq = np.sum((w - w_new) ** 2)
        w = w_new
        if chisq < tol:
            break
    w[np.abs(w) < tol] = 1e-15
    return w


def L2(A, y, log_lam=5.0):
    """L2 regularized least squares solver.

    Args:
        A (ndarray): The design matrix for the linear problem, shape ``(N, M)``.
        y (ndarray): The data vector, length ``N``.
        log_lam (float or ndarray, optional): The log of the regularization
            strength parameter, ``lambda``. This may either be a scalar or
            a vector of length ``N``. Defaults to 5.0.

    Returns:
        ndarray: The vector of coefficients ``w`` that minimizes the L2 norm
            for the linear problem.
    """
    ATA = A.T @ A
    ATy = A.T @ y
    ATA[np.diag_indices_from(ATA)] += 10 ** log_lam
    try:
        w = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(ATA, ATy, rcond=None)[0]
    return w


def lstsq(A, y):
    """Unregularized least squares solver.

    Args:
        A (ndarray): The design matrix for the linear problem, shape ``(N, M)``.
        y (ndarray): The data vector, length ``N``.

    Returns:
        ndarray: The vector of coefficients ``w`` that minimizes the chi squared
            loss.
    """
    ATA = A.T @ A
    ATy = A.T @ y
    try:
        w = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(ATA, ATy, rcond=None)[0]
    return w


def poly_design_matrix(x, ncol):
    """A design matrix constructed out of the Legendre polynomial basis.

    Args:
        x (ndarray): The independent coordinate vector, length ``N``.
        ncol (int): The number of columns ``M`` in the design matrix.
            Note that this corresponds to a polynomial fit with highest
            order equal to ``M - 1``.

    Returns:
        ndarray: The design matrix for the linear problem, shape ``(N, M)``.
    """
    return np.hstack(
        [legendre(n)(np.array(x)).reshape(-1, 1) for n in range(ncol)]
    )


def get_problem1_data():
    """Generate the dataset for Problem 1.

    Returns:
        pandas.DataFrame: A data frame containing the data vectors ``x`` and
            ``y`` as well as boolean masks ``train_idx`` and ``test_idx``
            specifying the indices of the training set points and test set
            points, respectively.
    """
    # Hard-coded settings
    npts = 100
    train_step = 10
    order = 50
    nfit = 1000
    err = 0.01

    # Generate the (x, y) data
    np.random.seed(42)
    x_ = np.linspace(-0.5, 0.5, nfit)
    y_ = x_ + np.exp(-(x_ ** 2) / 0.0025)
    A_ = poly_design_matrix(x_, order)
    w = L1(A_, y_, log_lam=0)
    x = np.linspace(-0.5, 0.5, npts)
    A = poly_design_matrix(x, order)
    y = A @ w
    y += err * np.random.randn(len(x))

    # Divvy up into train and test sets
    idx_train = np.zeros(npts, dtype=bool)
    idx_train[np.arange(0, npts, train_step)] = True
    idx_train[-1] = True
    idx_test = ~idx_train

    # Return
    df = pd.DataFrame(
        {"x": x, "y": y, "train_idx": idx_train, "test_idx": idx_test}
    )
    return df


def interact(
    data,
    plot_test_set=False,
    regularize="none",
):
    """Interactive widget for visualizing the effects of regularization.

    Args:
        data (pandas.DataFrame): The data frame for Problem 1; see
            ``get_problem1_data``.
        plot_test_set (bool, optional): Whether to show the terst set.
            Defaults to False.
        regularize (str, optional): Whether or not to apply regularization.
            Options are "none", "l1", and "l2". Defaults to "none".
    """
    # Hard-coded settings
    max_poly_order = 50

    # Get the data
    x = data["x"]
    y = data["y"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    # Log lambda range
    if str(regularize).lower() == "none":
        log_lam = [0]
    elif str(regularize).lower() == "l1":
        log_lam = [
            -18,
            -17.5,
            -17,
            -16.5,
            -16.375,
            -16.25,
            -16.125,
            -16,
            -15.9375,
            -15.875,
            -15.75,
            -15.5,
            -15.25,
            -15,
            -10,
            -2,
            -1.5,
            -1,
            -0.5,
            0,
            0.5,
            1,
            2,
            3,
        ]
    elif str(regularize).lower() == "l2":
        log_lam = [
            -18,
            -17.5,
            -17,
            -16.5,
            -16,
            -15.9375,
            -15.875,
            -15.75,
            -15.5,
            -15.25,
            -15,
            -10,
            -2,
            -1,
            -0.5,
            0,
            0.5,
            1,
            2,
            3,
        ]
    else:
        raise ValueError(f"Invalid regularizer: {regularize}")

    # Define our controls
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max_poly_order,
        step=1,
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=False,
        layout=Layout(width="90%"),
        description="poly order",
    )

    s_text = widgets.Label(value="{:.2f}".format(0))

    l_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(log_lam) - 1,
        step=1,
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=False,
        layout=Layout(width="90%"),
        description="log lambda",
    )

    l_text = widgets.Label(value="{:.2f}".format(log_lam[0]))

    def visualize_func(N=0, log_lam_idx=0):

        s_text.value = "{:d}".format(N)
        l_text.value = "{:.2f}".format(log_lam[log_lam_idx])

        # Compute the weights
        A_train = poly_design_matrix(x_train, N + 1)
        if str(regularize).lower() == "none":
            w = lstsq(A_train, y_train)
        elif str(regularize).lower() == "l1":
            w = L1(A_train, y_train, log_lam[log_lam_idx])
        elif str(regularize).lower() == "l2":
            w = L2(A_train, y_train, log_lam[log_lam_idx])
        else:
            raise ValueError(f"Invalid regularizer: {regularize}")
        model_train = A_train.dot(w)

        # Compute the prediction
        A_test = poly_design_matrix(x_test, N + 1)
        model_test = A_test.dot(w)

        # Compute the model on a high res grid
        x_hires = np.linspace(
            np.concatenate((x_train, x_test)).min(),
            np.concatenate((x_train, x_test)).max(),
            300,
        )
        A_hires = poly_design_matrix(x_hires, N + 1)
        model_hires = A_hires.dot(w)

        # Set up the plot
        fig = plt.figure(figsize=(15, 8))
        fig.subplots_adjust(wspace=0.25)
        ax = fig.subplot_mosaic(
            """
        AAB
        AAC
        """
        )
        ax["A"].set_xlabel("x", fontsize=28)
        ax["A"].set_ylabel("y(x)", fontsize=28)
        ax["A"].set_xlim(-0.5, 0.5)
        ymin = np.min(y_train)
        ymax = np.max(y_train)
        if plot_test_set:
            ymin = min((ymin, np.min(y_test)))
            ymax = max((ymax, np.max(y_test)))
        ypad = 0.5 * (ymax - ymin)
        ax["A"].set_ylim(ymin - ypad, ymax + ypad)

        # Plot the data
        ax["A"].plot(x_train, y_train, "ko")
        if plot_test_set:
            ax["A"].plot(x_test, y_test, "C1o")

        # Plot the model
        x = np.concatenate((x_train, x_test, x_hires))
        m = np.concatenate((model_train, model_test, model_hires))
        idx = np.argsort(x)
        x = x[idx]
        m = m[idx]
        ax["A"].plot(x, m, "C0-")

        # Print the loss
        loss_train = np.sum((y_train - model_train) ** 2) / len(y_train)
        ax["B"].text(
            0.1, 0.5, f"Train loss: {loss_train:.2e}", ha="left", fontsize=20
        )
        if plot_test_set:
            loss_test = np.sum((y_test - model_test) ** 2) / len(y_test)
            ax["B"].text(
                0.1,
                0.35,
                f"Test loss:  {loss_test:.2e}",
                ha="left",
                fontsize=20,
            )
        ax["B"].axis("off")

        # Plot the weights
        ax["C"].plot(np.log10(np.abs(w)), "C1-")
        ax["C"].plot(np.log10(np.abs(w)), "k.")
        ax["C"].axhline(0, color="k", lw=1, alpha=0.5, ls="--")
        ax["C"].set_xlim(0, max_poly_order)
        ax["C"].set_ylim(-15, 15)
        ax["C"].set_ylabel("log abs weights", fontsize=16)
        ax["C"].set_xlabel("weight index", fontsize=16)

    plot = interactive_output(
        visualize_func, {"N": slider, "log_lam_idx": l_slider}
    )

    # Display!
    display(plot)
    display(widgets.HBox([slider, s_text]))
    if str(regularize).lower() != "none":
        display(widgets.HBox([l_slider, l_text]))


def get_cv_loss(data, A, regularize="none", log_lam=0):
    """Returns the cross-validation loss in both the training set and the test set.

    Args:
        data (pandas.DataFrame): A dataframe containing the dataset,
            ``x`` and ``y``, and the training/test set boolean index masks,
            ``train_idx`` and ``test_idx``.
        A (ndarray): The design matrix for the linear problem, shape ``(N, M)``.
        regularize (str, optional): Whether or not to apply regularization. Options are
            "none", "l1", and "l2". Defaults to "none".
        log_lam (float or ndarray, optional): The log of the regularization
            strength parameter, ``lambda``. This may either be a scalar or
            a vector of length ``N``. Defaults to 5.0.

    Returns:
        tuple: Two floats corresponding to the loss in the training set and
            the test set, respectively.

    """
    A_train = A[data["train_idx"]]
    A_test = A[data["test_idx"]]
    if str(regularize).lower() == "none":
        w = lstsq(A_train, data["y"][data["train_idx"]])
    elif str(regularize).lower() == "l1":
        w = L1(A_train, data["y"][data["train_idx"]], log_lam=log_lam)
    elif str(regularize).lower() == "l2":
        w = L2(A_train, data["y"][data["train_idx"]], log_lam=log_lam)
    else:
        raise ValueError(f"Invalid regularizer: {regularize}")
    train_loss = (
        np.sum((A_train @ w - data["y"][data["train_idx"]]) ** 2)
        / A_train.shape[0]
    )
    test_loss = (
        np.sum((A_test @ w - data["y"][data["test_idx"]]) ** 2)
        / A_test.shape[0]
    )
    return train_loss, test_loss


def get_problem3_data():
    """Generate the dataset for Problem 3.

    Returns:
        pandas.DataFrame: A data frame containing the data vectors ``x`` and
            ``y`` as well as the housekeeping variable vectors ``temperature``,
            ``cloudiness``, ``psf_stability``, ``humidity``, and ``air_pressure``.
    """
    # Params
    np.random.seed(1)
    t = np.arange(0, 27, 1.0 / 24.0 / 60.0)
    nreg = 5
    log_S0 = 5.0
    log_Q = -2.0
    log_w0 = 0.0
    lam = 10.0
    flux_err = 1.00
    t0 = 17.0
    sig_t = 0.5

    # Value we want to recover
    depth_true = 8.50

    # Build the noise design matrix
    kernel = celerite.terms.SHOTerm(
        log_S0=log_S0, log_Q=log_Q, log_omega0=log_w0
    )
    gp = celerite.GP(kernel)
    gp.compute(t)
    A = gp.sample(size=nreg).T
    A -= np.mean(A, axis=0).reshape(1, -1)

    # Randomize a weight vector
    w_true = np.sqrt(lam) * np.random.randn(nreg)

    # The transit model
    A_trn = -np.exp(-((t - t0) ** 2) / sig_t ** 2).reshape(-1, 1)

    # Weight vector
    w_trn = [depth_true]

    # Add noise
    flux = (
        A.dot(w_true) + A_trn.dot(w_trn) + flux_err * np.random.randn(len(t))
    )

    # Return
    df = pd.DataFrame(
        {
            "x": t,
            "y": flux,
            "temperature": A[:, 0],
            "cloudiness": A[:, 1],
            "psf_stability": A[:, 2],
            "humidity": A[:, 3],
            "air_pressure": A[:, 4],
        }
    )
    return df


def get_problem4_data():
    """Generate the dataset for Problem 4.

    Returns:
        pandas.DataFrame: A data frame containing the data vectors ``x`` and
            ``y`` as well as 500 housekeeping vectors named ``A000`` through
            ``A499``.
    """
    # Params
    np.random.seed(0)
    t = np.arange(0, 27, 1.0 / 24.0 / 60.0)
    nreg = 500
    log_S0 = 7.0
    log_Q = -2.0
    log_w0 = 0.0
    lam = 1e-3
    flux_err = 1.0
    t0 = 12.0
    sig_t = 0.15

    # Value we want to recover
    depth_true = 6.0

    # Build the noise design matrix
    kernel = celerite.terms.SHOTerm(
        log_S0=log_S0, log_Q=log_Q, log_omega0=log_w0
    )
    gp = celerite.GP(kernel)
    gp.compute(t)
    A = gp.sample(size=nreg).T
    A -= np.mean(A, axis=0).reshape(1, -1)

    # Randomize a weight vector
    w_true = np.sqrt(lam) * np.random.randn(nreg)

    # The transit model
    A_trn = -np.exp(-((t - t0) ** 2) / sig_t ** 2).reshape(-1, 1)

    # Weight vector
    w_trn = [depth_true]

    # Add noise
    flux = (
        A.dot(w_true) + A_trn.dot(w_trn) + flux_err * np.random.randn(len(t))
    )

    # Return
    data = {"x": t, "y": flux}
    for n in range(nreg):
        data.update({"A%03d" % n: A[:, n]})
    df = pd.DataFrame(data)
    return df