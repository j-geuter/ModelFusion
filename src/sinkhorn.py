import torch
from logger import logging
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinkhorn(
    d1,
    d2,
    C,
    eps,
    max_iter=100,
    start=None,
    log=False,
    mcvThr=0,
    normThr=0,
    tens_type=torch.float32,
    verbose=True,
    min_start=None,
    max_start=None,
    show_progress_bar=True,
):
    """
    Sinkhorn's algorithm to compute the dual potentials and the dual problem value. Allows for parallelization.
    :param d1: source distribution(s). Two-dimensional tensor where first dimension corresponds to number of samples and second dimension to sample size. Can also be 1D for a single sample.
    :param d2: target distribution(s). Two-dimensional tensor as above.
    :param C: cost matrix. Two-dimensional tensor.
    :param eps: regularizer.
    :param max_iter: maximum number of iterations.
    :param start: first iteration's starting vector. If None, this is set to ones.
    :param log: if True, also computes the average marginal constraint violation and returns it.
    :param mcvThr: if greater than 0, the algorithm terminates if all approximations lie below this threshold, measured in terms of marginal constraint violation.
    :param normThr: terminates once all approximations lie within this threshold to their previous iterates, in terms of euclidean distance. Normalized by the norm at the start. 1e-3 to 1e-4 seem to be good choices.
    :param tens_type: determines the dtype of all tensors involved in computations.
    :param verbose: if False, turns off all info and warnings.
    :param min_start: if given, sets all entries in the starting vector smaller than `min_start` equal to `min_start`.
    :param max_start: if given, sets all entries in the starting vector larger than `max_start` equal to `max_start`.
    :param show_progress_bar: if True, shows a progress bar.
    :return: A dict with keys 'cost' (transport costs), 'plan' (transport plans), 'iterations' (number of iterations),
        'f' (first dual potential), 'g' (second dual potential),
        'u' (first scaling vector), 'v' (second scaling vector),
        'avgMCV' (average marginal constraint violations, only returned if `log`==True).
    """
    mu = d1.clone().to(device)
    nu = d2.clone().to(device)
    if mu.dim() == 1:
        mu = mu[None, :]
    if nu.dim() == 1:
        nu = nu[None, :]
    if not torch.all(torch.abs(mu.sum(dim=1) - 1) < 1e-6):
        logging.warning("d1 does not sum to 1! Rescaled to probability measure.")
        mu /= mu.sum(dim=1).unsqueeze(1)
    if not torch.all(torch.abs(nu.sum(dim=1) - 1) < 1e-6):
        logging.warning("d2 does not sum to 1! Rescaled to probability measure.")
        nu /= nu.sum(dim=1).unsqueeze(1)
    if start is None:
        start = torch.ones(nu.size())
    start = start.detach().to(device)
    if max_start:
        start = torch.where(
            start < torch.tensor(max_start).to(start.dtype).to(device),
            start,
            torch.tensor(max_start).to(start.dtype).to(device),
        )
    if min_start:
        start = torch.where(
            start > torch.tensor(min_start).to(start.dtype).to(device),
            start,
            torch.tensor(min_start).to(start.dtype).to(device),
        )
    mu = mu.T.to(tens_type)
    nu = nu.T.to(tens_type)
    start = start.T.to(tens_type)
    K = torch.exp(-C / eps).to(tens_type).to(device)
    v = start.clone()
    it = max_iter
    iterable = range(max_iter) if not show_progress_bar else tqdm(range(max_iter))
    for i in iterable:
        prev_v = v.clone()
        u = mu / torch.matmul(K, v)
        v = nu / torch.matmul(K.T, u)
        if normThr > 0:
            if torch.all(
                (torch.norm(prev_v - v, dim=0) / torch.norm(v, dim=0)) < normThr
            ):
                it = i + 1
                if verbose:
                    logging.info(
                        f"Accuracy below threshold. Early termination after {it} iterations."
                    )
                break
        if mcvThr > 0:
            gamma = torch.matmul(
                torch.cat(
                    [torch.diag(u.T[j]).to(device)[None, :] for j in range(u.size(1))]
                ),
                torch.matmul(
                    K,
                    torch.cat(
                        [
                            torch.diag(v.T[j]).to(device)[None, :]
                            for j in range(u.size(1))
                        ]
                    ),
                ),
            )
            mu_star = torch.matmul(
                gamma, torch.ones(u.size(0)).to(tens_type).to(device)
            )
            nu_star = torch.matmul(
                torch.ones(u.size(0)).to(tens_type).to(device), gamma
            )
            mu_err = (mu.T - mu_star).abs().sum(1)
            nu_err = (nu.T - nu_star).abs().sum(1)
            if (mu_err < mcvThr).sum().item() == mu_err.size(0) and (
                nu_err < mcvThr
            ).sum().item() == nu_err.size(0):
                it = i + 1
                if verbose:
                    logging.info(
                        f"Accuracy below threshold. Early termination after {it} iterations."
                    )
                break
    gamma = torch.matmul(
        torch.cat([torch.diag(u.T[j]).to(device)[None, :] for j in range(u.size(1))]),
        torch.matmul(
            K,
            torch.cat(
                [torch.diag(v.T[j]).to(device)[None, :] for j in range(u.size(1))]
            ),
        ),
    )
    cost = (gamma * C).sum(1).sum(1)
    perc_nan = 100 * cost.isnan().sum() / len(cost)
    if perc_nan > 0:
        perc_nan = "%.2f" % perc_nan
        logging.warning(f"{perc_nan}% of transport costs are NaN.")
    if not log:
        return {
            "cost": cost,
            "plan": gamma.squeeze(),
            "iterations": it,
            "f": eps * torch.log(u).T,
            "g": eps * torch.log(v).T,
            "u": u.T,
            "v": v.T,
        }
    else:
        mu_star = torch.matmul(gamma, torch.ones(u.size(0)).to(tens_type).to(device))
        nu_star = torch.matmul(torch.ones(u.size(0)).to(tens_type).to(device), gamma)

        mu_err = (mu.T - mu_star).abs().sum(1)
        mu_nan = mu_err.isnan().sum()
        mu_err = torch.where(
            mu_err.isnan(), torch.tensor(0).to(tens_type).to(device), mu_err
        )

        nu_err = (nu.T - nu_star).abs().sum(1)
        nu_nan = nu_err.isnan().sum()
        nu_err = torch.where(
            nu_err.isnan(), torch.tensor(0).to(tens_type).to(device), nu_err
        )

        if mu_nan / mu_err.size(0) > 0.1 or nu_nan / nu_err.size(0) > 0.1:
            perc1 = 100 * mu_nan / mu_err.size(0)
            perc2 = 100 * nu_nan / nu_err.size(0)
            perc = "%.2f" % ((perc1 + perc2) / 2)
            if verbose:
                logging.warning(f"{perc}% of marginal constraint violations are NaN.")

        mu_err = mu_err.sum() / (mu_err.size(0) - mu_nan)
        nu_err = nu_err.sum() / (nu_err.size(0) - nu_nan)
        return {
            "cost": cost,
            "plan": gamma.squeeze(),
            "iterations": it,
            "f": eps * torch.log(u).T,
            "g": eps * torch.log(v).T,
            "u": u.T,
            "v": v.T,
            "avgMCV": (mu_err + nu_err) / 2,
        }


def log_sinkhorn(
    mu,
    nu,
    C,
    eps,
    max_iter=100,
    start_f=None,
    start_g=None,
    log=False,
    tens_type=torch.float32,
):
    """
    Sinkhorn's algorithm in log domain to compute the dual potentials and the dual problem value.
    :param mu: first distribution. One-dimensional tensor. Also supports two-dimensional tensor with an empty first dimension.
    :param nu: second distribution. One-dimensional tensor as above.
    :param C: cost matrix. Two-dimensional tensor.
    :param eps: regularizer.
    :param max_iter: maximum number of iterations.
    :param start: first iteration's starting vector. If None, this is set to ones.
    :param log: if True, returns the optimal plan and dual potentials alongside the cost; otherwise, returns only the cost.
    :param tens_type: determines the dtype of all tensors involved in computations. Defaults to float64 as this allows for greater accuracy.
    :return: if log==False: the transport cost. Else: a dict with keys 'cost' (transport cost), 'plan' (transport plan), 'u' and 'v' (dual potentials).
    """

    def S(cost, pot1, pot2):
        """
        Auxiliary function for log_sinkhorn.
        """
        ones = torch.ones(pot1.size())[None, :].to(pot1.dtype).to(device)
        return (
            cost
            - torch.matmul(pot1[None, :].T, ones)
            - torch.matmul(ones.T, pot2[None, :])
        )

    def row_min(A, eps):
        """
        Auxiliary function for log_sinkhorn.
        """
        return -eps * torch.log(torch.exp(-A / eps).sum(1))

    if mu.dim() == 2:
        mu = mu.view(-1)
    if nu.dim() == 2:
        nu = nu.view(-1)
    mu = mu.to(tens_type).to(device)
    nu = nu.to(tens_type).to(device)
    if start_f == None:
        start_f = torch.zeros(mu.size())
    if start_g == None:
        start_g = torch.ones(mu.size())
    start_f = start_f.detach().to(tens_type).to(device)
    start_g = start_g.detach().to(tens_type).to(device)
    f = start_f
    g = start_g
    for i in range(max_iter):
        f = row_min(S(C, f, g), eps) + f + eps * torch.log(mu)
        g = (
            row_min(S(C, f, g).T, eps) + g + eps * torch.log(nu)
        )  # the column minimum function is equivalent to the row minimum function of the transpose
    gamma = torch.matmul(
        torch.diag(torch.exp(f / eps)).to(device),
        torch.matmul(
            torch.exp(-C.to(tens_type) / eps), torch.diag(torch.exp(g / eps)).to(device)
        ),
    )
    cost = (gamma * C.to(tens_type)).sum().item()
    if not log:
        return cost, gamma.squeeze()
    else:
        return {"cost": cost, "plan": gamma.squeeze(), "u": f.T, "v": g.T}
