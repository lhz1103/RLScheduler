class EmaNormalizer:
    """
    Exponential-moving-average z-score normalizer.
    Keeps running (μ, σ) with decay ρ, then returns z = (x-μ)/(σ+ε).

    Parameters
    ----------
    rho : float, default 0.999
        Smoothing factor. ρ≈0.999 ≈ 1⁄(1-ρ)=1 000 时隙。
    eps : float, default 1e-6
        Lower bound to keep σ away from zero.
    warmup_steps : int, default 100
        Number of initial samples that bypass normalisation
        (σ is unreliable while counts are small).
    """

    __slots__ = ("rho", "eps", "warmup_steps", "_mu", "_nu", "_count")

    def __init__(self, rho: float = 0.97, eps: float = 1e-6, warmup_steps: int = 5):
        self.rho = rho
        self.eps = eps
        self.warmup_steps = warmup_steps
        self._mu = 0.0      # running EMA mean
        self._nu = 0.0      # running EMA second moment
        self._count = 0     # number of samples seen

    # ------------------------------------------------------------------ #
    def update(self, x: float) -> float:
        """
        Feed one raw scalar `x`, return its z-score under current EMA stats.
        """
        self._count += 1

        # --- 1. 更新一阶与二阶矩 -------------------------------------- #
        one_minus_rho = 1.0 - self.rho
        self._mu = self.rho * self._mu + one_minus_rho * x
        self._nu = self.rho * self._nu + one_minus_rho * (x * x)

        # --- 2. 计算方差与标准差 -------------------------------------- #
        var = self._nu - self._mu * self._mu
        sigma = (var if var > 0.0 else 0.0) ** 0.5
        sigma = max(sigma, self.eps)

        # --- 3. 冷启动阶段：直接返回 0 -------------------------------- #
        #if self._count <= self.warmup_steps:
            #return x

        # --- 4. z-score 标准化 --------------------------------------- #
        return (x - self._mu) / sigma

    # ------------------------------------------------------------------ #
    @property
    def mean(self) -> float:
        return self._mu

    @property
    def std(self) -> float:
        return max((self._nu - self._mu * self._mu) ** 0.5, self.eps)
