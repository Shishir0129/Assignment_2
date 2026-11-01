from scipy.stats import beta as beta_dist


def posterior_update(k, n, alpha=300, beta=700):
    """Update a Beta(alpha, beta) prior with k successes (sand) in n trials.

    previous prior: Beta(197,803)
    new preior: Beta(300,700)

    Returns the posterior parameters, mean, mode and 95% credible interval.
    """
    assert alpha > 0 and beta > 0, "alpha and beta must be > 0"
    assert 0 <= k <= n, "k must be between 0 and n"

    a_post = alpha + k                 # alpha + number of sand samples
    b_post = beta + n - k              # beta  + number of shale samples

    mean = a_post / (a_post + b_post)
    if a_post > 1 and b_post > 1:
        mode = (a_post - 1) / (a_post + b_post - 2)
    else:
        mode = None

    ci_low, ci_high = beta_dist.ppf([0.025, 0.975], a_post, b_post)

    return {"alpha": a_post, "beta": b_post, "mean": mean,
            "mode": mode, "ci95": (ci_low, ci_high)}


result = posterior_update(k=197, n=1000)   # data: 197 sand of 1000; posterior Beta(497, 1503)
print(result)
