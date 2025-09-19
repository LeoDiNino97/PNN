import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.decomposition import PCA


def compute_covariance_and_precision(X, prec_type, lambda_, eta):
    """
    Initialize the covariance and precision matrix based
    on the settings of the current experiments
    """
    N = X.shape[1]
    if X.shape[2] > 1:
        Xcov = X.permute(0,2,1).reshape(-1, N).cpu()
    else:
        Xcov = X.squeeze().T.cpu()

    if prec_type in ["sample", "joint", "naive", "cov"]:
        C = torch.cov(Xcov)
        Prec = torch.linalg.inv(C)

    elif prec_type == "glasso":
        C = torch.cov(Xcov)
        Prec = graphical_lasso_with_reg(C, torch.zeros([Xcov.shape[0], Xcov.shape[0]]), gamma=0, tau=lambda_*eta, eta=eta)
        C, Prec = torch.FloatTensor(C).to(Xcov.device), torch.FloatTensor(Prec).to(Xcov.device)

    else:
        raise NotImplementedError(f"Covariance {prec_type} not available")
 
    return C, Prec


def compute_metric(output, target, metric):
    if metric == "accuracy":
        return compute_multiclass_accuracy(output, target)
    elif metric == "roc_auc":
        preds = output.argmax(1).type_as(target)
        return roc_auc_score(target.cpu(), preds.cpu())


def compute_multiclass_accuracy(output, target):
    preds = output.argmax(1).type_as(target)
    correct = preds.eq(target).double()
    correct = correct.sum()
    return correct / len(target)


def sparse_cov_loss(gamma, C, sampleC, lam):
    return gamma * ( torch.log(torch.det(C)) + torch.trace(torch.inverse(C) @ sampleC ) + lam * C.abs().sum())



def make_psd(A):
    """
    Projects a symmetric matrix A onto the PSD cone by zeroing out negative eigenvalues.
    
    Args:
        A (torch.Tensor): A symmetric matrix of shape (N, N)
        eps (float): Minimum eigenvalue threshold to ensure numerical stability

    Returns:
        torch.Tensor: PSD projection of A
    """
    # Ensure symmetry
    A_sym = 0.5 * (A + A.T)

    # Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(A_sym)

    # Clip negative eigenvalues to 0
    eigvals_clipped = torch.clamp(eigvals, min=1e-6)

    # Reconstruct PSD matrix
    A_psd = eigvecs @ torch.diag(eigvals_clipped).to(A_sym.device) @ eigvecs.T
    return A_psd


def compute_loss(Loss, prec_type, y_true, y_pred, gamma, Theta_tilde, Theta, sampleC=None, lam=0.0):
    loss = Loss(y_pred, y_true) 
    if prec_type == "joint":
        loss += gamma * torch.linalg.norm(Theta_tilde - Theta, 'fro')
    elif prec_type == "naive":
        loss += - torch.log(torch.linalg.det(Theta_tilde)) + torch.trace(sampleC @ Theta_tilde) + lam * torch.abs(Theta_tilde).sum()

    return loss



def soft_threshold(X, tau):
    """
    Applies soft thresholding to the off-diagonal elements of X, leaving the diagonal untouched.

    Args:
        X (torch.Tensor): Input square tensor
        tau (float): Threshold value

    Returns:
        torch.Tensor: Tensor with off-diagonal elements soft-thresholded
    """
    assert X.shape[0] == X.shape[1], "Input must be a square matrix"
    
    mask = 1 - torch.eye(X.shape[0], device=X.device)  # 1 for off-diagonal, 0 for diagonal
    X_thresh = torch.sign(X) * torch.clamp(X.abs() - tau, min=0.0)
    return X * (1 - mask) + X_thresh * mask


def graphical_lasso_with_reg(C_hat, Theta_tilde, gamma, eta=.01, cur_Theta=None, eps=0.0, tau=0.0075, max_iter=10000, tol=1e-8, lam=0.1):
    """
    Compute graphical lasso with L2 regularization toward a target matrix Theta_tilde.

    Params:
        C_hat (torch.Tensor): Empirical covariance matrix (N x N)
        Theta_tilde (torch.Tensor): Target precision matrix (N x N)
        gamma (float): Regularization strength
        eta (float): Step size for gradient descent
        cur_Theta (torch.Tensor, optional): Current estimate of the precision matrix
        eps (float): Small value to ensure numerical stability (diagonal loading for inverse)
        tau (float): Threshold for soft thresholding
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance
    """

    if cur_Theta is None:
        # Theta = torch.linalg.inv(C_hat + eps * torch.eye(C_hat.shape[0], device=C_hat.device))
        Theta = torch.eye(C_hat.shape[0], device=C_hat.device)
    else:
        Theta = cur_Theta

    for iteration in range(max_iter):
        Theta_old = Theta.clone()
        grad = -torch.linalg.inv(Theta + eps * torch.eye(Theta.shape[0], device=C_hat.device)) + C_hat + 2 * gamma * (Theta - Theta_tilde)
        Theta = Theta - eta * grad
        Theta = soft_threshold(Theta, tau)
        Theta = make_psd(Theta)

        # Convergence check
        delta = torch.linalg.norm(Theta - Theta_old, ord='fro') / torch.linalg.norm(Theta_old, ord='fro')
        if delta < tol:
            break
    
    return Theta


# Generate synthetic task with sparse precision
def generate_task(n_samples, n_features, SNR, alpha, seed=1):

    prng = np.random.RandomState(seed)
    prec = make_sparse_spd_matrix(
        n_features, alpha=alpha, smallest_coef=0.4, largest_coef=0.7, random_state=prng
    )
    cov = np.linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    # Generate labels by shifting signal over clean precision
    y = X

    w = np.random.rand(n_features)
    y = y @ w
    if SNR is not None:
        signal_power = np.mean(y ** 2)
        snr_linear = 10 ** (SNR / 10.0)
        noise_power = signal_power / snr_linear
        noise = prng.normal(scale=np.sqrt(noise_power), size=y.shape)
        y += noise

    return X, y, cov, prec


def perform_pca(X, K, pca=None):
    """
    If pca=None, computes the covariance eigenvectors and performs PCA,
    otherwise it applies the transformation in pca.
    """

    T, N, F = X.shape
    X_reshaped = X.transpose(0, 2, 1).reshape(-1, N)  # (T*F, N)

    # Fit PCA
    if pca:
        X_reduced = pca.transform(X_reshaped)  # (T*F, K)
    else:
        pca = PCA(n_components=K)
        X_reduced = pca.fit_transform(X_reshaped)  # (T*F, K)

    X_out = X_reduced.reshape(T, F, K).transpose(0, 2, 1)  # (T, K, F)
    return X_out, pca
