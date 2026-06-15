# Compute-Adaptive Risk Control: Distribution-Free Joint Guarantees for Early-Exit Networks

*Draft prepared for submission to Transactions on Machine Learning Research (TMLR).*

---

## Abstract

Early-exit (multi-exit) networks reduce inference cost by letting "easy" inputs leave at shallow exits, but the deployed exit policy is typically tuned heuristically on a validation set with **no formal guarantee** that the realized error stays below a target — and we show this naive tuning silently violates the target at a rate far above any nominal level. We give a calibration procedure that selects exit confidence thresholds so the deployed cascade satisfies a user-specified risk constraint $R(\tau)\le\alpha$ with finite-sample, distribution-free validity, at a compute cost close to that of the (invalid) naive policy. We build on the Learn-then-Test (LTT) reduction of risk control to multiple hypothesis testing; we propose no new test, and are precise about scope. *Validity* — the deployed policy keeps true risk below the target — is finite-sample, distribution-free, and training-conditional (it holds for the deployed $\hat\tau$, not merely on average over calibration draws). On top of validity we add: (i) a **cost characterization** of the fixed-sequence test on the compute-ordered chain of exit policies — the certified policy is never unsafe and never undercuts the safe frontier, and under a monotone-risk condition its excess compute over the cheapest safe policy *in the chain* is bounded by a boundary band of width $O(\sqrt{\ln(K/\delta)/n})$, so the $\ln K$ price of the search falls on *efficiency*, never validity; (ii) a joint risk–compute variant that certifies a *pre-committed* compute budget for settings (provisioning, latency SLAs) where the budget must be guaranteed before the test distribution is seen; and (iii) a **shift-robust certificate under estimated importance weights**, which we present as a conservative sufficient condition / sensitivity tool: sample splitting plus an $L^1$ weight-error budget $\eta$ keep validity for any deployment whose weight error is at most $\eta$, at a feasibility cost we characterize. We give explicit concentration p-values and prove every guarantee used. Empirically, we verify the $1-\delta$ guarantee and the naive-tuning failure in a controlled early-exit simulation (where ground-truth risk is available for violation counting) and in a CIFAR-100 branchy-CNN cache diagnostic (where held-out test-pool risk is the finite-pool proxy); the remaining ImageNet, language-cascade, tabular, and shift experiments are specified as protocol.

---

## 1. Introduction

Adaptive-depth inference — early-exit networks, cascades, and routers — is now a standard tool for trading accuracy against latency and energy. The dominant deployment recipe is: train a multi-exit model, then pick confidence thresholds on held-out data so that the average accuracy looks acceptable and the average compute looks affordable. This recipe has two well-known gaps. First, "looks acceptable on a validation set" is not a guarantee: the threshold is chosen by inspecting the same statistic it is supposed to control, so the realized test risk can exceed the target, and the failure rate is not quantified. Second, accuracy and compute are tuned jointly by hand, with no principled statement about *both* quantities at once.

This paper asks a narrow, testable question: **can we choose exit thresholds so that the deployed policy provably keeps a chosen risk below a target $\alpha$ with probability at least $1-\delta$, distribution-free and finite-sample, while spending as little compute as possible?** We answer yes by instantiating the Learn-then-Test (LTT) framework of Angelopoulos et al. for the structure of exit policies. The testing-to-risk-control reduction and the fixed-sequence/graphical tests for monotone hypotheses are LTT's. What this paper adds is (a) a *cost characterization* of the fixed-sequence test on the compute-ordered chain — it is safe by construction, and we bound how much compute the search costs relative to the cheapest safe policy in the chain; (b) a joint risk–compute certificate for pre-committed budgets; and (c) a shift-robust certificate that stays finite-sample valid under *estimated* importance weights via sample splitting and an $L^1$ sensitivity budget.

The central claim is a guarantee, not a state-of-the-art accuracy number, so the experiments are built to *falsify* the guarantee if it is wrong — by measuring violation frequency across many calibration/test splits — and to quantify the (small) compute the guarantee costs over the invalid naive policy. The practical takeaway we want a reader to leave with is concrete: **confidence-threshold tuning by validation inspection does not control risk, and a distribution-free certificate that does is available at almost the same compute.**

**Scope of evidence.** To set expectations precisely: the guarantees in §§3–6 are *proved* (Appendix A); the $1-\delta$ behavior and the naive-tuning failure are *verified in a controlled simulation* with a reference implementation, where ground-truth risk is available for violation counting (§10.1); and one real-model cache diagnostic has been run on a branchy CIFAR-100 CNN (§10.2), using held-out test-pool risk as a finite-pool proxy rather than population truth. The ImageNet, language-cascade, tabular, and real shift studies remain planned. We are careful below not to state planned findings as established.

### 1.1 Contributions

The reduction "risk control = multiple testing" and the admissibility of fixed-sequence and graphical tests are due to LTT and the multiple-testing literature. On that basis:

1. **Formulation.** We cast exit-threshold selection for an $L$-exit network as risk control over a finite, *compute-ordered* family of configurations, for any bounded loss $\ell_{\mathrm{loss}}\in[0,1]$ (0–1 error, selective risk, class-conditional false-negative rate). The relevant structural fact is that the deployment objective — the cheapest safe policy — and the compute order of the chain coincide, so LTT's fixed-sequence test applies with the order fixed *a priori* rather than chosen post hoc.
2. **Cost characterization of the chain test (Theorem 2).** Validity holds unconditionally and with no chain-length penalty (this is LTT/fixed-sequence). The new statement is two-sided: the certified policy is always safe and never undercuts the safe frontier, and under a monotone-risk condition it lands within a boundary band of the cheapest safe policy *in the chain*, with excess compute bounded by the compute spanned by configurations whose true risk lies in $(\alpha-\Delta_n,\alpha]$, $\Delta_n=O(\sqrt{\ln(K/\delta)/n})$. The $\ln K$ cost of searching the chain thus falls on *efficiency*, not validity.
3. **Joint risk–compute control (Theorem 3).** We certify $R(\tau)\le\alpha$ and $C(\tau)\le B$ simultaneously with probability $\ge1-\delta$. This matters when the budget must be *pre-committed* — hardware provisioning, energy caps, latency SLAs — and so cannot be checked against the test distribution after the fact (unlike realized compute, which is observable at deployment). The risk certificate is the essential one; the compute certificate is a convenience for pre-commitment, and we treat it as such.
4. **Shift-robust certificate under estimated weights (Theorem 4) — a conservative sufficient condition.** Risk control under covariate shift is usually stated with *exact* importance weights. We give a finite-sample valid version under *estimated* weights by splitting off the weight-fitting fold and testing against a deflated target $\alpha-\eta$, where $\eta$ bounds the $L^1$ weight error. We do **not** claim this is tight or broadly practical: it is a sufficient condition whose validity rests on an unverifiable budget $\eta$, and whose feasibility degrades as $\eta$ grows. Its honest reading is a sensitivity analysis — "the target holds for any deployment whose weight error is at most $\eta$" — and we position it alongside the sensitivity-analysis line in conformal prediction, not as new machinery.
5. **Reference implementation and empirical checks.** A unit-tested implementation of all p-values, certification procedures, and selectors, with a controlled-simulation study (§10.1) verifying the $1-\delta$ guarantee, the naive-tuning failure, and the chain/Holm/Bonferroni cost ordering; plus a CIFAR-100 branchy-CNN cache diagnostic (§10.2) and a specified protocol for the remaining real-data studies.

### 1.2 Relation to Learn-then-Test

*Borrowed.* From LTT (Angelopoulos, Bates, Candès, Jordan, Lei) and the multiple-testing literature: the reduction of risk control to testing nulls $H_\tau:R(\tau)>\alpha$ with valid p-values; the admissibility of fixed-sequence and graphical FWER procedures for monotone/nested hypotheses; the Hoeffding–Bentkus and empirical-Bernstein p-values; and the fact that any selection rule restricted to the certified set inherits validity.

*New.* The cost characterization of the chain test for exit policies (Theorem 2b); the joint risk–compute certificate for pre-committed budgets (Theorem 3); the estimated-weight shift sufficient condition (Theorem 4); a reference implementation with controlled-simulation and CIFAR-100 cache diagnostics of validity and naive-tuning failure, plus a specified protocol for the remaining real-data studies. The contribution is a deployment methodology with provable guarantees plus one (conservative) shift result, not new testing machinery. For the assumption-free fallback we implement and evaluate **Holm's step-down**; the general graphical family (Bretz–Maurer–Brannath) subsumes both the chain and Holm and can recycle significance along the compute order, but we do not separately implement or evaluate it.

---

## 2. Problem setup and notation

Let $(X,Y)\sim P$ with $X\in\mathcal X$, $Y\in\mathcal Y$. A multi-exit network has exits $\ell\in\{1,\dots,L\}$ ordered by depth. Exit $\ell$ produces a prediction $\hat y_\ell(x)$ and a scalar confidence $s_\ell(x)\in[0,1]$ (e.g., top softmax probability, or $1-$ entropy). Let $c_\ell\ge0$ be the cumulative compute (FLOPs, or wall-clock proxy) to *reach and evaluate* exit $\ell$, with $c_1<c_2<\dots<c_L=:C_{\max}$.

**Exit policy.** A configuration is a threshold vector $\tau=(\tau_1,\dots,\tau_{L-1})\in[0,1]^{L-1}$. On input $x$, the policy exits at

$$
E_\tau(x)=\min\big(\{\,\ell<L : s_\ell(x)\ge\tau_\ell\,\}\cup\{L\}\big),
$$

i.e., it stops at the first exit whose confidence clears its threshold, and otherwise runs to the final exit $L$. The deployed prediction is $\hat Y_\tau(x)=\hat y_{E_\tau(x)}(x)$.

**Risk and compute functionals.** For a bounded loss $\ell_{\mathrm{loss}}:\mathcal Y\times\mathcal Y\to[0,1]$,

$$
R(\tau) \;=\; \mathbb E_{P}\!\big[\ell_{\mathrm{loss}}(\hat Y_\tau(X),Y)\big],
\qquad
C(\tau) \;=\; \mathbb E_{P}\!\big[c_{E_\tau(X)}\big].
$$

Examples of $\ell_{\mathrm{loss}}$: 0–1 error $\mathbf 1\{\hat Y_\tau\ne Y\}$; selective risk on accepted points; class-conditional false-negative rate (clip to $[0,1]$). All that the theory needs is boundedness.

**Calibration data.** We observe $n$ i.i.d. draws $\mathcal D=\{(X_i,Y_i)\}_{i=1}^n\sim P$, disjoint from training. Define empirical estimates

$$
\hat R(\tau)=\frac1n\sum_{i=1}^n \ell_{\mathrm{loss}}(\hat Y_\tau(X_i),Y_i),
\qquad
\hat C(\tau)=\frac1n\sum_{i=1}^n c_{E_\tau(X_i)}.
$$

**Goal.** Given target risk $\alpha\in(0,1)$ and confidence level $\delta\in(0,1)$, return $\hat\tau$ (a measurable function of $\mathcal D$) such that

$$
\Pr_{\mathcal D}\big(R(\hat\tau)\le\alpha\big)\;\ge\;1-\delta,
\tag{P1}
$$

and, among configurations satisfying (P1), $\hat\tau$ has small compute $C(\hat\tau)$. A stronger *dual* specification adds a budget $B$:

$$
\Pr_{\mathcal D}\big(R(\hat\tau)\le\alpha \ \text{and}\ C(\hat\tau)\le B\big)\ge 1-\delta.
\tag{P2}
$$

**Grid.** We work over a finite candidate set $\Lambda\subset[0,1]^{L-1}$ (e.g., a per-exit grid of $m$ thresholds, $|\Lambda|\le m^{L-1}$, or a scalar-parameterized chain; see §4). All probabilities below are over the draw of $\mathcal D$; $\hat\tau$ is data-dependent.

**Nature of the guarantee.** (P1) is a *training-conditional* (PAC-style) statement: with probability $\ge1-\delta$ over the calibration draw, the *fixed, deployed* policy $\hat\tau$ has true risk $\le\alpha$ for all subsequent inputs. It is not a marginal-over-future-test-points guarantee that averages away calibration noise; it is the stronger object an operator wants, since $\hat\tau$ is frozen at deployment. The qualifier *distribution-free* applies to **validity**: it holds for any $P$. It does **not** extend to the efficiency (cost-optimality) and shift results, which require, respectively, a monotonicity condition (§4.1) and a weight-error budget (§6.2); we flag this wherever it matters.

---

## 3. Risk control as multiple testing

We adapt Learn-then-Test (LTT). For each $\tau\in\Lambda$ define the null

$$
H_\tau:\quad R(\tau)>\alpha \quad(\text{``}\tau\text{ is unsafe''}).
$$

Rejecting $H_\tau$ certifies $\tau$ as safe. We need (a) a valid p-value $p_\tau$ for each $H_\tau$, and (b) a family-wise-error-controlling (FWER) procedure that maps $\{p_\tau\}$ to a certified set $\hat\Lambda\subseteq\Lambda$ with

$$
\Pr\big(\exists\,\tau\in\hat\Lambda:\ R(\tau)>\alpha\big)\le\delta.
\tag{1}
$$

Given (1), *any* selection rule restricted to $\hat\Lambda$ inherits validity, because the bad event "we output an unsafe $\tau$" is contained in "$\hat\Lambda$ contains an unsafe $\tau$."

### 3.1 Valid p-values from concentration

Because $\ell_{\mathrm{loss}}\in[0,1]$ and the $n$ losses are i.i.d. given $\tau$, $\hat R(\tau)$ is a mean of bounded i.i.d. variables. The one-sided Hoeffding inequality gives, for the boundary mean $R(\tau)=\alpha$ and any observed $\hat R(\tau)\le\alpha$,

$$
\Pr_{R(\tau)=\alpha}\!\big(\hat R(\tau)\le \hat r\big)\;\le\;\exp\!\big(-2n(\alpha-\hat r)^2\big).
$$

Since under $H_\tau$ we have $R(\tau)\ge\alpha$ (taking the closure), the supremum of the left side over the null is attained at $R(\tau)=\alpha$. Hence

$$
\boxed{\,p_\tau \;=\; \exp\!\big(-2n\,(\alpha-\hat R(\tau))_+^2\big)\,}
\tag{2}
$$

is a valid (super-uniform under $H_\tau$) p-value, where $(z)_+=\max(z,0)$. Tighter choices — the **Hoeffding–Bentkus (HB)** p-value used in LTT, or an empirical-Bernstein p-value when the loss has small variance — slot in directly and improve power; we use (2) for exposition and HB in experiments.

### 3.2 Bonferroni LTT (assumption-free baseline)

Reject $H_\tau$ iff $p_\tau\le\delta/|\Lambda|$, giving $\hat\Lambda_{\mathrm{Bonf}}=\{\tau:p_\tau\le\delta/|\Lambda|\}$. Then output

$$
\hat\tau=\arg\min_{\tau\in\hat\Lambda_{\mathrm{Bonf}}}\hat C(\tau)
\quad(\text{abstain to }\tau=\mathbf 1\ \text{i.e. always run to }L\ \text{if }\hat\Lambda_{\mathrm{Bonf}}=\varnothing).
$$

**Theorem 1 (Distribution-free risk validity).** *For losses in $[0,1]$, i.i.d. calibration data, p-values (2), and the Bonferroni rule at level $\delta$, the output $\hat\tau$ satisfies (P1): $\Pr(R(\hat\tau)\le\alpha)\ge1-\delta$.*

*Proof.* By Bonferroni, $\Pr(\exists\tau\in\hat\Lambda_{\mathrm{Bonf}}:R(\tau)>\alpha)\le\sum_{\tau:R(\tau)>\alpha}\Pr(p_\tau\le\delta/|\Lambda|)\le |\{\tau:H_\tau\}|\cdot \delta/|\Lambda|\le\delta$, using super-uniformity of $p_\tau$ under $H_\tau$. On the complementary event every certified $\tau$ is safe; since $\hat\tau\in\hat\Lambda_{\mathrm{Bonf}}$ (or is the safe full-depth policy with $R=R(\mathbf 1)\le\alpha$ whenever $\alpha\ge R(\mathbf1)$; if $\alpha<R(\mathbf1)$ the task is infeasible and we report infeasibility), $R(\hat\tau)\le\alpha$. $\square$

The defect of Bonferroni is the $\delta/|\Lambda|$ penalty: with a fine per-exit grid $|\Lambda|$ is large, the threshold is tiny, and the certified set is small and expensive. The next section avoids this penalty on the *validity* side by testing a pre-ordered chain — at the price of restricting the search to that chain, a trade-off we make explicit.

---

## 4. The compute-chain instantiation

Fixed-sequence testing is a standard FWER procedure, already named in LTT for monotone, pre-orderable hypotheses; this section does **not** propose it as new. What it contributes is that early-exit deployment supplies the pre-specified order without extra design, aligned with the deployment objective, plus a two-sided cost bound (Theorem 2b) on how close the resulting policy is to the cheapest safe policy in the chain.

### 4.1 The monotone structure of exit policies

Order configurations by **compute**. Consider a *chain* of configurations $\tau^{(1)}\preceq\tau^{(2)}\preceq\dots\preceq\tau^{(K)}$ that is increasing in compute, i.e. $C(\tau^{(1)})\le\dots\le C(\tau^{(K)})$, with $\tau^{(K)}=\mathbf 1$ the always-run-to-$L$ policy. A natural and common construction is a **scalar-parameterized** family $\tau(t)=(t,\dots,t)$ for $t$ on a grid $0=t_1<\dots<t_K=1$: raising the global threshold $t$ makes every exit more conservative, pushing more inputs to deeper exits, which *increases* compute monotonically (deterministically, since each input's exit index is nondecreasing in $t$). More refined chains (per-exit thresholds raised in a fixed schedule) are also admissible.

**Assumption M (monotone risk along the chain).** $R(\tau^{(1)})\ge R(\tau^{(2)})\ge\dots\ge R(\tau^{(K)})$; equivalently, deeper-on-average policies have weakly lower risk.

Assumption M says deeper exits are, on average, at least as accurate — the premise that motivates deploying a deep model at all. Its empirical version (monotone $\hat R$ along the chain) is observable, and we report it; the assumption itself concerns the true $R$. Crucially, Assumption M is needed only for the *efficiency* statement of Theorem 2(b), never for validity.

**Within-chain scope.** A chain is a one-dimensional curve through the $(L{-}1)$-dimensional configuration space, and the scalar family $\tau(t)=(t,\dots,t)$ in particular is restrictive: per-exit thresholds can dominate a single shared threshold. Consequently "cheapest safe policy" in Theorem 2 means cheapest *along the chosen chain*, not the global optimum over all of $[0,1]^{L-1}$. The chain is a deliberate design choice trading search breadth against the multiplicity penalty: a Holm step-down over the full grid recovers more configurations at a $\log|\Lambda|$ cost (§4.2, remark), and the more general graphical family is a possible extension. We report the chain and Holm variants and let the practitioner pick between those implemented procedures.

### 4.2 Certification along the chain

Apply LTT's fixed-sequence test to the compute chain, walking from most expensive to cheapest, $\tau^{(K)},\tau^{(K-1)},\dots$, and **stopping at the first configuration that fails to certify**. With the Hoeffding p-value (2), the certification test $p_{\tau^{(k)}}\le\delta$ is exactly
$$
\hat R(\tau^{(k)})\;\le\;\alpha-\epsilon_n(\delta),
\qquad
\epsilon_n(\delta):=\sqrt{\tfrac{\ln(1/\delta)}{2n}} .
$$
Let $k^\star$ be the cheapest index reached before stopping; output $\hat\tau=\tau^{(k^\star)}$. If even $\tau^{(K)}$ fails to certify, report the task infeasible at this $(\alpha,\delta,n)$ and deploy $\tau^{(K)}=\mathbf 1$.

Define the cheapest truly-safe index $k_0:=\min\{k:R(\tau^{(k)})\le\alpha\}$.

**Theorem 2 (validity and a two-sided cost bound).**
*(a) Validity (LTT/fixed-sequence; stated for completeness).* For p-values valid under each $H_{\tau^{(k)}}$ and the fixed-sequence rule at level $\delta$ on the pre-specified chain order, $\Pr(R(\hat\tau)\le\alpha)\ge1-\delta$, with no dependence on $K$.
*(b) Cost bound (our contribution).* On the validity event the output never undercuts the safe frontier: $C(\hat\tau)\ge\min_{k:R(\tau^{(k)})\le\alpha}C(\tau^{(k)})$, and under Assumption M this means $k^\star\ge k_0$. For the other direction, let
$$
\Delta_n:=\epsilon_n(\delta)+\epsilon_n(\delta/K)=O\!\Big(\sqrt{\tfrac{\ln(K/\delta)}{n}}\Big),
\qquad
k_1:=\min\{k:\ R(\tau^{(j)})\le\alpha-\Delta_n\ \text{for all}\ j\ge k\}.
$$
Then $\Pr(k^\star\le k_1)\ge1-\delta$. Combining, on an event of probability $\ge1-2\delta$,
$$
k_0\ \le\ k^\star\ \le\ k_1,
$$
so the excess compute of the certified policy over the cheapest safe policy in the chain is at most $C(\tau^{(k_1)})-C(\tau^{(k_0)})$ — the compute spanned by configurations whose true risk lies in the boundary band $(\alpha-\Delta_n,\,\alpha]$. As $n\to\infty$, $\Delta_n\to0$ and $k_1\to k_0$.

*Proof.* (a) is the standard fixed-sequence guarantee: since the order is fixed in advance and we stop at the first non-rejection, an error (outputting an unsafe config) requires rejecting the first true null in the walk, which has probability $\le\delta$ by super-uniformity; this is independent of $K$.

(b) *Lower bound.* $\hat\tau$ is certified, hence safe on the validity event, so its compute is at least that of the cheapest safe configuration; under Assumption M the safe set is the suffix $\{k\ge k_0\}$ and compute is increasing in $k$, giving $k^\star\ge k_0$. *Upper bound.* The walk reaches index $k_1$ iff every config in $\{k_1,\dots,K\}$ certifies, i.e. $\hat R(\tau^{(j)})\le\alpha-\epsilon_n(\delta)$ for all $j\ge k_1$. For any such $j$, $R(\tau^{(j)})\le\alpha-\Delta_n=\alpha-\epsilon_n(\delta)-\epsilon_n(\delta/K)$, so
$$
\Pr\big(\hat R(\tau^{(j)})>\alpha-\epsilon_n(\delta)\big)\le\Pr\big(\hat R(\tau^{(j)})-R(\tau^{(j)})>\epsilon_n(\delta/K)\big)\le e^{-2n\,\epsilon_n(\delta/K)^2}=\delta/K
$$
by the upper-tail Hoeffding inequality. A union bound over the at most $K$ such indices gives $\Pr(\exists j\ge k_1:\ \hat R(\tau^{(j)})>\alpha-\epsilon_n(\delta))\le\delta$, i.e. $\Pr(k^\star\le k_1)\ge1-\delta$. Intersecting with the validity event and a union bound yields the sandwich with probability $\ge1-2\delta$. $\square$

The structure of the bound is the honest payoff: **validity carries no $K$-penalty** (part a), while the **price of searching a length-$K$ chain is an $O(\sqrt{\ln K/n})$ band on efficiency** (part b). This is why early stopping matters and why we do *not* claim exact recovery of $k_0$: a safe configuration sitting within $\Delta_n$ of the boundary may fail to certify and halt the walk, returning a slightly more expensive policy — but never an unsafe one. Versus Bonferroni over the same grid, the chain's per-test level is $\delta$ rather than $\delta/K$, which tightens $\epsilon_n(\delta)$ and shrinks the band; the gain is LTT's, realized here because the compute order supplies the chain for free.

**How large is the chain-vs-Bonferroni gain, in practice?** Honestly, often small. The advantage enters only through the *difference* between $\epsilon_n(\delta)$ and $\epsilon_n(\delta/K)$, i.e. $\sqrt{\ln(1/\delta)/2n}$ vs. $\sqrt{\ln(K/\delta)/2n}$, which is modest unless $K$ is large or $n$ small (in our controlled simulation, §10.1, the certified-compute gap between chain and Bonferroni is a few percent). The robust reading of this paper is therefore *validity at near-naive compute*, not large savings over Bonferroni: the naive policy is cheap but invalid; the chain is valid at almost the naive cost, and slightly cheaper than Bonferroni, with the gap widening as $K$ grows and $\delta$ tightens.

> **Remark (assumption-free fallback when Assumption M is doubtful).** When per-exit thresholds are tuned jointly, or the empirical-monotonicity diagnostic looks suspect, drop the chain for an FWER procedure that needs no ordering. We implement and evaluate **Holm's step-down**, which is valid unconditionally, requires no monotonicity, and strictly dominates Bonferroni. The general *graphical* family (Bretz–Maurer–Brannath) subsumes both the chain and Holm and can recycle significance along the compute partial order — reducing to the chain test under exact monotonicity at a $\log|\Lambda|$ band — but we do not implement or evaluate it here, and do not claim its specific behavior.

---

## 5. Joint risk–compute control

Problem (P2) additionally certifies a compute budget $B$. The motivation is pre-commitment: an operator who must provision hardware, cap energy, or sign a latency SLA needs the budget guaranteed *before* the test distribution is seen, whereas realized compute can simply be measured afterward. So this is a convenience guarantee layered on the essential risk one, not a co-equal claim. Compute $c_{E_\tau(X)}\in[0,C_{\max}]$ is bounded, so $\hat C(\tau)$ concentrates by Hoeffding: for a single $\tau$, with probability $\ge1-\delta_C$,
$$
C(\tau)\;\le\;\hat C(\tau)+C_{\max}\sqrt{\tfrac{\ln(1/\delta_C)}{2n}}=:U_C(\tau;\delta_C).
$$
To hold uniformly over the data-dependent choice, take a union bound over the $K$ chain points (or $|\Lambda|$ grid points), replacing $\delta_C$ by the *deterministic* $\delta_C/K$ (resp. $\delta_C/|\Lambda|$).

**Algorithm (dual control).** Split $\delta=\delta_R+\delta_C$. (i) Certify risk: form $\hat\Lambda$ as in §3–§4 at level $\delta_R$. (ii) Restrict to budget-feasible configs using the *uniform* compute bound: $\hat\Lambda_B=\{\tau\in\hat\Lambda: U_C(\tau;\delta_C/K)\le B\}$. (iii) Output $\hat\tau=\arg\min_{\tau\in\hat\Lambda_B}\hat C(\tau)$, or report infeasible if $\hat\Lambda_B=\varnothing$.

**Theorem 3 (Joint guarantee).** *With the split budget $\delta=\delta_R+\delta_C$ and the procedure above,*
$$
\Pr\big(R(\hat\tau)\le\alpha \ \text{and}\ C(\hat\tau)\le B\big)\ge1-\delta.
$$

*Proof.* Let $\mathcal A=\{\forall\tau\in\hat\Lambda:R(\tau)\le\alpha\}$ and $\mathcal B=\{\forall\tau\in\Lambda:C(\tau)\le U_C(\tau;\delta_C/K)\}$ (the compute event ranges over the *fixed* grid, with the deterministic correction $\delta_C/K$). By §3–§4, $\Pr(\mathcal A^c)\le\delta_R$; by Hoeffding and a union bound over the $K$ points, $\Pr(\mathcal B^c)\le\delta_C$. On $\mathcal A\cap\mathcal B$: $\hat\tau\in\hat\Lambda\Rightarrow R(\hat\tau)\le\alpha$, and $\hat\tau\in\hat\Lambda_B\Rightarrow C(\hat\tau)\le U_C(\hat\tau;\delta_C/K)\le B$. Union bound: $\Pr((\mathcal A\cap\mathcal B)^c)\le\delta_R+\delta_C=\delta$. $\square$

A symmetric formulation — *minimize risk subject to a hard budget* — swaps the roles, certifying a budget by inverting the compute bound and minimizing $\hat R$ over budget-feasible configs; both are special cases of certifying one functional and optimizing the other.

**On the split $(\delta_R,\delta_C)$.** We do not present this as a contribution. Because $\delta_R,\delta_C$ enter only through $\sqrt{\ln(1/\cdot)}$, the split is a weak knob: moving mass between the two sides changes each slack only logarithmically. We default to $\delta_R=0.9\delta,\ \delta_C=0.1\delta$ — favoring the risk side, which is the essential guarantee and whose extra power directly buys cheaper certified policies — and verify in E5 that performance is flat across a broad central range. An operator who genuinely needs to optimize it faces a trivial one-dimensional search.

**Looseness of the compute bound.** The range-$C_{\max}$ Hoeffding UCB $U_C$ is conservative, since per-input cost is far from uniform on $[0,C_{\max}]$; in our simulation the dual certificate is reported infeasible whenever the budget $B$ is set near the realized compute, even when a budget-respecting safe policy exists. An empirical-Bernstein compute bound (variance-adaptive, as in §6.1) tightens $U_C$ materially and is the recommended form in practice; we keep Hoeffding in the statement for transparency. This is a further reason we treat the compute certificate as secondary to the risk certificate.

---

## 6. Covariate-shift robustness

Suppose deployment is under $Q$ with $X\sim Q_X$, $Y\mid X$ unchanged, and likelihood ratio $w(x)=\mathrm dQ_X/\mathrm dP_X$. The target is $R_Q(\tau)=\mathbb E_Q[\ell_{\mathrm{loss}}]=\mathbb E_P[w(X)\,\ell_{\mathrm{loss}}]$. The exact-weight case (§6.1) is standard; §6.2 handles the realistic case where $w$ is estimated.

### 6.1 Exact weights (warm-up)

If $0\le w\le W$ is known exactly, then $w\,\ell_{\mathrm{loss}}\in[0,W]$ has mean $R_Q(\tau)$, and the §3.1 argument rescaled by range $W$ gives the valid p-value $p^{Q,\mathrm{exact}}_\tau=\exp(-\frac{2n}{W^2}(\alpha-\hat R_Q(\tau))_+^2)$ with $\hat R_Q(\tau)=\frac1n\sum_i w(X_i)\ell_i$. Theorems 1–2 then hold under $Q$. Two caveats matter in practice. (i) Hoeffding with range $W$ is *loose* for importance weighting because it ignores that most weights are small; the empirical-Bernstein p-value, whose slack scales with the *weighted variance* rather than the range, is materially tighter and is what we use in experiments. (ii) The often-quoted effective sample size $n_{\mathrm{eff}}\approx(\sum_i w_i)^2/\sum_i w_i^2$ describes the behavior of that variance-adaptive bound, *not* of the range-$W$ Hoeffding bound above; we report $n_{\mathrm{eff}}$ only alongside the empirical-Bernstein certificate, to avoid conflating the two.

### 6.2 Estimated weights

Let weights be **estimated** by any method (density-ratio estimation, a discriminative classifier, moment matching). Two problems arise: (a) the estimator $\hat w$ is a function of data, so treating it as fixed in a Hoeffding bound is invalid; (b) $\hat w\ne w$ introduces bias. We resolve (a) by **sample splitting** and (b) by an **$L^1$ error budget**.

Split the calibration set into independent folds $\mathcal D_1$ (size $n_1$) and $\mathcal D_2$ (size $n_2$). Estimate $\hat w$ on $\mathcal D_1$ *only*. Suppose the estimate is bounded, $0\le\hat w\le\hat W$, and obeys an $L^1$ error budget
$$
\mathbb E_P\big[\,|\hat w(X)-w(X)|\,\big]\;\le\;\eta
\qquad(\text{conditionally on }\mathcal D_1).
\tag{$\star$}
$$
Compute the weighted empirical risk on the held-out fold, $\hat R_Q(\tau)=\frac1{n_2}\sum_{i\in\mathcal D_2}\hat w(X_i)\,\ell_i$, and test against the **deflated** target $\alpha-\eta$:
$$
\boxed{\,p^{\,Q}_\tau \;=\; \exp\!\Big(-\tfrac{2 n_2}{\hat W^2}\,\big((\alpha-\eta)-\hat R_Q(\tau)\big)_+^2\Big)\,}.
\tag{3}
$$

**Theorem 4 (Shift-robust validity under estimated weights).** *Assume ($\star$), $0\le\hat w\le\hat W$, $\eta<\alpha$, and $Y\mid X$ invariant across $P,Q$. Then the p-value (3) is super-uniform under $H^Q_\tau:R_Q(\tau)>\alpha$, and the chain (or Bonferroni) certification of §3–§4 applied to $\{p^Q_\tau\}$ at level $\delta$ yields*
$$
\Pr\big(R_Q(\hat\tau)\le\alpha\big)\ge1-\delta\qquad(\text{under }Q).
$$

*Proof.* Condition on $\mathcal D_1$; then $\hat w$ is a fixed (deterministic) function and the summands $\hat w(X_i)\ell_i$ for $i\in\mathcal D_2$ are i.i.d. in $[0,\hat W]$ with common mean $\mu:=\mathbb E_P[\hat w\,\ell_{\mathrm{loss}}]$. Decompose
$$
\mu=\mathbb E_P[w\,\ell_{\mathrm{loss}}]+\mathbb E_P[(\hat w-w)\ell_{\mathrm{loss}}]=R_Q(\tau)+\mathbb E_P[(\hat w-w)\ell_{\mathrm{loss}}].
$$
Since $\ell_{\mathrm{loss}}\in[0,1]$, the bias term is bounded by $|\mathbb E_P[(\hat w-w)\ell_{\mathrm{loss}}]|\le\mathbb E_P|\hat w-w|\le\eta$ by ($\star$). Under $H^Q_\tau$ we have $R_Q(\tau)>\alpha$, hence $\mu\ge R_Q(\tau)-\eta>\alpha-\eta$. By one-sided Hoeffding with range $\hat W$, for any observed $\hat R_Q(\tau)=r\le\alpha-\eta$,
$$
\Pr\big(\hat R_Q(\tau)\le r\,\big|\,\mathcal D_1\big)\le\exp\!\Big(-\tfrac{2n_2}{\hat W^2}(\mu-r)^2\Big)\le\exp\!\Big(-\tfrac{2n_2}{\hat W^2}\big((\alpha-\eta)-r\big)^2\Big)=p^Q_\tau.
$$
So $p^Q_\tau$ is super-uniform under $H^Q_\tau$ conditionally on $\mathcal D_1$; since this holds for every realization of $\mathcal D_1$, it holds marginally. The downstream FWER arguments (Theorems 1–2) are unchanged, run conditionally on $\mathcal D_1$ and then marginalized, giving the stated $1-\delta$ validity under $Q$. $\square$

**Reading the result.** The three costs are explicit. *(i)* Data splitting shrinks the testing fold to $n_2<n$; we take $n_1$ as small as the weight estimator tolerates, since weight estimation is typically more sample-efficient than the Hoeffding slack on the testing fold. *(ii)* The deflation $\alpha\to\alpha-\eta$ requires $\eta<\alpha$ and makes the certificate conservative in proportion to the weight-error budget. *(iii)* Range inflation $\hat W$ — mitigated by using the empirical-Bernstein form of (3), exactly as in §6.1. The budget $\eta$ has two sources: a theoretical $L^1$ rate for the chosen density-ratio estimator under stated smoothness, or — more defensibly, since it needs no knowledge of $w$ — a sensitivity reading: the certificate is valid for any deployment whose weight error is at most $\eta$, so the practitioner reports the largest $\eta^\star$ (the most severe shift) under which the target still holds.

**Positioning.** The two ingredients are individually standard: conditioning on a held-out fold to make a plug-in estimator a fixed function, and absorbing estimation error into a margin in the spirit of sensitivity analysis (cf. the $\Gamma$-style robustness bounds and estimated-weight treatments in the conformal-under-shift literature). The contribution is their combination in the LTT risk-control setting, where we have not seen the estimated-weight case stated with a finite-sample proof. We claim validity and interpretability, not tightness.

**When this is useful, and when it is not.** Two honest limits, both visible in §10.1. First, $\eta$ must genuinely upper-bound the estimator's $L^1$ error for validity to hold; it cannot be read off the data. When $\eta$ is set large enough to be defensible, the deflation $\alpha-\eta$ can leave little room and the certificate becomes *infeasible* — in our simulation feasibility falls toward zero once $\eta$ approaches a fifth of $\alpha$. So the result buys robustness at a feasibility price that is only acceptable when the achievable risk sits well below $\alpha$. Second, validity when $\eta$ is *under*-set is not automatic: it held in our simulation only because the fitted density-ratio happened to be conservatively biased; an anti-conservatively biased estimator with $\mathbb E|\hat w-w|>\eta$ can and does breach the target (we exhibit this in §10.1 with a deliberately biased estimator). The takeaway is to treat Theorem 4 as a sensitivity statement with a known feasibility cost, not as a turnkey shift fix.

---

## 7. Algorithm summary

```
Input: multi-exit model with exits 1..L; calibration data D (n points);
       target risk α; confidence δ; chain {τ^(1) ⪯ ... ⪯ τ^(K)=1} (incr. compute);
       optional budget B; optional shift {weight estimator, L1-budget η}; split (δ_R, δ_C).
0. (Shift only) split D into D1, D2; fit ŵ on D1; set test fold = D2, target = α−η, range = Ŵ.
   (No shift) test fold = D, target = α, range = 1.
1. For each τ^(k): on the test fold compute exit index E_{τ^(k)}(X_i), prediction, loss ℓ_i,
   cost c_{E}(X_i); form  R̂(τ^(k))  (weighted R̂_Q with ŵ if shifted)  and  Ĉ(τ^(k)).
2. p-values: p_k ← Eq.(2) / HB / weighted Eq.(3), at the target and range from step 0.
3. Risk certification (choose one):
     - Fixed-sequence (chain): walk k = K,K-1,...; certify while p_k ≤ δ_R; stop at first p_k > δ_R.
     - Holm: step-down over a finite grid when the chain order is doubtful (assumption-free fallback used here).
     - Bonferroni: certify all k with p_k ≤ δ_R/K   (assumption-free baseline).
   → certified set  Λ̂.
4. (Dual) Budget filter: Λ̂_B ← { k ∈ Λ̂ : Ĉ(τ^(k)) + C_max·sqrt(ln(K/δ_C)/(2 n_test)) ≤ B },
   with (δ_R, δ_C) a fixed split (default 0.9δ, 0.1δ; the split is a weak, logarithmic knob).
5. Output τ̂ ← argmin over Λ̂ (or Λ̂_B) of Ĉ;  else report infeasible / fall back to τ=1.
Guarantee: P( R(τ̂) ≤ α  [and C(τ̂) ≤ B] ) ≥ 1 − δ   (under Q when the shift branch is used).
```

Cost: one forward pass collecting all exit scores per calibration point ($O(n)$ inference), then $O(nK)$ to score the chain — negligible relative to training.

---

## 8. Related work

- **Conformal prediction / risk control.** Split conformal (Vovk; Lei et al.); conformal risk control (Angelopoulos et al.); Learn-then-Test (Angelopoulos, Bates, Candès, Jordan, Lei) — our backbone. We do **not** claim the testing reduction, the fixed-sequence/graphical tests, or the HB p-value; these are LTT's. We contribute the early-exit instantiation with a two-sided cost bound (Thm 2b), the joint risk–compute certificate for pre-committed budgets (Thm 3), and the estimated-weight shift result (Thm 4).
- **Multiple testing.** Bonferroni; fixed-sequence / fallback procedures (Wiens); Holm step-down; and graphical sequentially-rejective tests (Bretz, Maurer, Brannath). We use Bonferroni, fixed-sequence, and Holm as off-the-shelf FWER tools. Graphical procedures are a more general adjacent family, but we do not implement or evaluate them here. The contribution is matching the compute order to a deployment objective, not the tests themselves.
- **Adaptive computation.** BranchyNet, MSDNet, SkipNet, patience-based exiting, LLM cascades and routers. These tune thresholds without distribution-free guarantees; we add the certificate and show (E3) that naive validation tuning violates the target above rate $\delta$.
- **Distribution shift.** Weighted conformal prediction (Tibshirani et al.); covariate-shift risk control; sensitivity-analysis conformal under unknown or misspecified weights (e.g. $\Gamma$-selection / bounded-odds bounds, Jin & Candès and related). Theorem 4 sits in this lineage: it combines sample splitting with an $L^1$ weight-error margin to give a finite-sample risk-control certificate under *estimated* weights, which we have not seen stated in the LTT setting.
- **Selective prediction.** Selective risk and coverage (Geifman & El-Yaniv); our $\ell_{\mathrm{loss}}$ can encode selective risk, unifying abstention with early exit.

---

## 9. Implementation plan

**Backbones and exits.**
- *Vision:* a completed branchy ResNet-56-style cache on CIFAR-100; planned MSDNet and early-exit ResNet-50 on ImageNet. Confidence $s_\ell$ = top softmax probability, with an entropy-based alternative planned.
- *Language:* a size cascade (e.g., a small encoder → medium → large) acting as a 3-exit policy on a text-classification benchmark; confidence from softmax margin. This stresses the chain with very uneven $c_\ell$.
- *Tabular:* a multi-exit MLP / gradient-boosted cascade on standard tabular benchmarks, where compute savings are mundane but the guarantee is easy to audit and the i.i.d. assumption is clean.

**Losses to certify.** (i) 0–1 error; (ii) selective risk with an abstain option folded into the policy; (iii) class-conditional false-negative rate on a designated positive class (clipped to $[0,1]$), to demonstrate non-symmetric risks.

**p-values.** Hoeffding (Eq. 2) and Hoeffding–Bentkus; empirical-Bernstein when loss variance is small. Report certified-set size and selected compute for each.

**Chains.** Default scalar chain $\tau(t)=(t,\dots,t)$, $K=100$ grid points; an alternative per-exit schedule chain; and a Holm step-down over a 2-D per-exit grid as the assumption-free ablation.

**Engineering.** Single calibration pass caching per-point, per-exit (score, prediction, correctness, cumulative FLOPs). FLOPs measured with a profiler; wall-clock reported separately as it is hardware-dependent and not the certified quantity. Released code: a thin library that takes cached calibration tensors + $(\alpha,\delta,B)$ and returns $\hat\tau$ and the certificate.

---

## 10. Experiments

We separate what is *done* from what is *planned*. §10.1 reports a controlled-simulation study that has been run with the released reference implementation; its purpose is to verify that the procedure behaves as the theorems predict in a setting where ground-truth risk is available for violation counting. §10.2 reports a completed CIFAR-100 cache diagnostic and then lists the remaining real-data protocol.

### 10.1 Controlled-simulation verification (done)

We use a synthetic early-exit model ($L=5$ exits, competence increasing with depth, noisy confidences, global-threshold chain of $K=40$ configurations) whose true per-configuration risk and cost are computable by large-sample Monte Carlo, so every selected policy can be checked against the truth. This is a sanity check on the mathematics and implementation, *not* evidence about real networks; it cannot validate Assumption M or weight estimability in practice (see §10.2). The reported values are written by the reproducibility scripts to `results/e1_validity.json`, `results/e2_chain_vs_bonferroni.json`, `results/e5_joint_control.json`, and `results/e6_shift_eta_sweep.json`.

*Validity and naive failure (E1/E3).* Across a grid of targets and $T=3000$ calibration/test splits per cell, the certified chain violates $R(\hat\tau)\le\alpha$ at rates of $0.0010$–$0.0143$, uniformly at or below $\delta\in\{0.05,0.10\}$. The naive recipe (cheapest config with $\hat R\le\alpha$, no correction) violates at $0.2313$–$0.3513$, several-fold above the nominal level. This is the paper's central empirical point: naive threshold tuning silently fails, and the certificate fixes it for a small compute premium.

*Cost of the guarantee (E2).* At a representative $(\alpha,\delta)=(0.119,0.10)$ with $T=2000$, the certified expected cost is $3.260$ (chain), $3.311$ (Holm), and $3.326$ (Bonferroni), all with true violation $\le\delta$. At the matched E1 target, the naive invalid policy costs $3.183$, so the chain pays a small compute premium for validity. The chain's advantage over Bonferroni is real but small ($\approx2\%$ here) and, consistent with Theorem 2(b), widens with $K$ and with smaller $\delta$. The honest summary is *validity at near-naive cost*, with a modest additional saving from exploiting the chain order.

*Joint control (E5).* With $\alpha=0.089$, $\delta=0.10$, $n=2000$, and $T=1000$, a slack budget $B=4.641$ is certified feasible in every trial, with no true risk violations and mean certified compute UCB $3.620$. A tight budget $B=3.363$ is oracle-feasible — the cheapest truly safe configuration costs $3.313$ — but the range-$C_{\max}$ compute certificate reports infeasible in every trial. This quantifies the conservativeness flagged in §5.

*Covariate shift (E6).* Under an exp-tilt shift toward harder inputs, the unweighted certificate (calibrated on source) violates the target at rate $0.875$. The estimated-weight sweep has realized mean weight error $\|\hat w-w\|_1\approx0.060$: for underspecified budgets $\eta\in\{0,0.03\}$ the certificate remains feasible with $0$ observed violations but lacks the assumption needed for validity; near the boundary ($\eta=0.06$) feasibility drops to $0.35$; and once $\eta$ clearly covers the mean error ($\eta\in\{0.09,0.12\}$), feasibility collapses to $0$. With a deliberately downward-biased estimator and $\eta=0.02$ too small ($\|\hat w-w\|_1\approx0.351$), the certificate breaches the target at rate $0.895\gg\delta$. The result supports the intended sensitivity reading of Theorem 4: an honestly sized $\eta$ can make the certificate vacuous, while an undersized or biased weight model can invalidate the target.

### 10.2 CIFAR-100 real-cache diagnostic and remaining protocol

We implemented the real-cache interface on a branchy ResNet-56-style CIFAR-100 model with four exits (after the stem and after each of three residual stages), scalar top-softmax thresholds on a $K=100$ grid, 0–1 loss, and analytical cumulative MAC estimates. The model was trained for 80 epochs and evaluated on the CIFAR-100 test set, producing `cache/cifar_branchy_resnet56.npz` and the cache-evaluation artifacts `results/cifar_branchy_real_cache_eval.json` and `results/cifar_branchy_real_cache_eval_exploratory.json`. The per-exit test accuracies are $0.0646$, $0.3040$, $0.6147$, and $0.6945$; the corresponding cumulative exit costs are $0.44$, $42.91$, $84.33$, and $125.75$ million MACs. Because this is a finite test-pool diagnostic, the reported violation frequencies use held-out split risk as a proxy for population risk, not as an oracle truth measurement.

**Pre-registered CIFAR grid.** The frozen CIFAR E1 grid uses $\alpha\in\{0.05,0.10,0.20,0.30\}$, $\delta\in\{0.05,0.10\}$, calibration sizes $\{500,2000\}$, Hoeffding–Bentkus p-values, and $T=1000$ random calibration/test partitions per cell. This run revealed a negative feasibility result: the full test-pool minimum risk along the chain is $0.3055$, so none of the pre-registered targets has a truly safe configuration. The chain therefore mostly abstains from selection. Its violation frequency over all split attempts is $0$ for $\alpha\le0.20$ and $0.000$–$0.013$ at $\alpha=0.30$, below both tested $\delta$ values, but feasibility at $\alpha=0.30$ is only $0.000$–$0.013$. The naive selector, by contrast, selects in $0.281$–$0.457$ of the $\alpha=0.30$ split attempts and violates in exactly those attempts, since the selected finite-pool policies remain above the target. This is useful evidence for the safety/abstention behavior and the naive failure mode, but not evidence that the current CIFAR model supports low-error certified deployment at the pre-registered targets.

**Exploratory feasible-target check.** To confirm that the cache evaluator behaves sensibly once the target is feasible, we ran a post hoc diagnostic at $\alpha\in\{0.35,0.40\}$ with the same $\delta$, calibration sizes, p-values, and split count. The chain violation frequencies are $0.012$–$0.038$, always below the corresponding $\delta$; naive validation tuning violates at $0.441$–$0.529$. Chain feasibility is high for $\alpha=0.40$ ($0.994$–$1.000$) and mixed for $\alpha=0.35$ ($0.484$–$0.996$), matching the fact that the feasible frontier is close to the final-exit risk. Mean chain cost is $85.65$–$105.30$ million MACs across these feasible-target cells, compared with $81.28$–$92.85$ million MACs for the invalid naive policy. Holm and Bonferroni are more conservative: they have lower or zero violation rates but lower feasibility and higher mean cost in the same cells.

**Monotonicity diagnostic.** The full-pool chain risk is not perfectly nonincreasing: four adjacent threshold pairs show risk increases, with maximum increase $2\times10^{-4}$, while cost is nondecreasing. This small empirical violation does not affect validity, but it means the Assumption M efficiency interpretation should be read as a diagnostic rather than a confirmed model property on this cache.

The remaining real-data study has not been run; it will use the same violation-frequency design.

**Backbones.** *Vision:* MSDNet / early-exit ResNet-50 on ImageNet, confidence = top softmax probability (and an entropy variant). *Language:* a size cascade (small→medium→large) as a 3-exit policy on text classification, which stresses the chain with very uneven $c_\ell$. *Tabular:* a multi-exit MLP on standard benchmarks, where the i.i.d. assumption is cleanest and the certificate is easiest to audit.

**E1 — Validity.** $T\ge1000$ splits per cell; report empirical violation frequency with binomial CIs against $\delta$; sweep $\alpha\in\{0.02,0.05,0.1,0.2\}$, $\delta\in\{0.05,0.1\}$. Pass: frequency $\le\delta$ within CI in every cell.
**E2 — Cost.** Risk–compute Pareto curves and exit-index histograms; chain vs. Holm vs. Bonferroni; report the realized chain-vs-Bonferroni gap rather than assuming it is large.
**E3 — Baselines.** Naive validation tuning, BranchyNet entropy thresholds, patience exiting, full-depth; the load-bearing comparison is naive (invalid, cheap) vs. ours (valid, slightly costlier).
**E4 — Calibration size and the band.** Vary $n$; show certified compute approaching the cheapest safe in-chain policy with the gap tracking $\Delta_n=O(\sqrt{\ln(K/\delta)/n})$. Report empirical monotonicity of $\hat R$ along the chain as a *diagnostic* (necessary, not sufficient, for Assumption M), with simultaneous CIs on the $R_k$ differences; report cases where it is violated and how Holm behaves there.
**E5 — Joint control.** Verify the joint event at rate $\ge1-\delta$; compare the Hoeffding vs. empirical-Bernstein compute bound (the latter expected to be far less often infeasible).
**E6 — Shift.** ImageNet-C corruptions and a label-shifted tabular set. Arms: unweighted (expected to violate), exact-weight oracle (efficiency ceiling on synthetic shifts), estimated-weight. Report the $\eta$-sensitivity curve (validity and feasibility vs. $\eta$), the realized $\|\hat w-w\|_1$ where a held-out estimate of it is possible, the split cost ($n_1/n_2$), and the failure mode under a biased estimator.
**E7 — Ablations.** p-value (Hoeffding vs. HB vs. empirical-Bernstein); chain granularity $K$; scalar vs. per-exit chain; chain vs. Holm.

**Reporting.** Every figure pairs realized violation frequency with compute, so the guarantee and its price are seen together; negative results (monotonicity-diagnostic failures, shift too severe for a non-vacuous certificate) are reported explicitly.

---

## 11. Limitations

- Validity rests on **i.i.d. calibration data** (or, under shift, an $L^1$ weight-error budget $\eta$ via Theorem 4); under unmodeled or unbounded shift the certificate can fail.
- Bounds use **bounded losses**; unbounded or heavy-tailed risks need different concentration (e.g., truncation or empirical-Bernstein with care).
- **Cost-optimality is within the chosen chain**, a one-dimensional slice of the configuration space; the global optimum over per-exit thresholds is not certified. Holm widens the implemented search at a $\log|\Lambda|$ efficiency cost but still does not exhaust the continuum; graphical procedures are a possible generalization, not an evaluated component of this paper.
- The compute certificate treats $c_\ell$ as **fixed FLOPs**; wall-clock latency under batching, caching, and hardware effects is *not* the certified quantity and is reported only descriptively. The compute certificate is also the *secondary* guarantee — useful for pre-commitment, but realized compute is observable at deployment in a way realized risk is not, and the range-$C_{\max}$ bound is loose enough to be infeasible when the budget is tight.
- **Assumption M governs efficiency, not validity**; when it fails the returned policy is still safe but may be needlessly expensive, and we fall back to Holm. The monotonicity *diagnostic* we report (monotone $\hat R$) is necessary but not sufficient for the true-risk condition.
- **Current real-model evidence is narrow.** The completed CIFAR-100 cache is a diagnostic on a modest branchy CNN, not a full real-data benchmark suite. Its pre-registered target grid is mostly infeasible because the final-exit test risk is $0.3055$; the feasible-target results are explicitly exploratory.
- **Theorem 4 is a conservative sufficient condition.** It is valid only when the unverifiable budget $(\star)$ holds; when $\eta$ is set large enough to be defensible the deflation can make the certificate infeasible, and when $\eta$ is under-set the target can be breached (both shown in §10.1). We give no tight (bias-corrected) version; obtaining one without sacrificing finite-sample validity is open. The budget $\eta$ must be supplied as an estimator rate or read as a sensitivity margin.
- **The guarantee is per single use.** Validity is training-conditional for *one* run of the procedure; a user who sweeps several $(\alpha,\delta,\text{chain})$ on the same calibration set and selects the most favorable outcome reintroduces multiplicity and forfeits the guarantee. Fix $(\alpha,\delta)$ and the chain before looking at the calibration risks.

---

## 12. Broader impact statement

This work is intended to make adaptive inference safer to deploy by replacing heuristic threshold tuning with an explicit finite-sample risk certificate. The main positive impact is operational: practitioners can state and audit when an early-exit policy satisfies a target risk rather than relying on validation-set inspection. The main risk is over-trust. The guarantee is only as good as its calibration assumptions, its single-use protocol, and, under shift, the supplied weight-error budget. A deployed system that ignores these conditions could still violate the target while appearing certified. We mitigate this risk in the paper by stating the assumptions at the point of use, reporting infeasibility rather than hiding it, and requiring pre-registration of $(\alpha,\delta)$, the chain, and the split protocol before calibration results are inspected.

---

## 13. Conclusion

Compute-adaptive models need threshold-selection procedures whose guarantees survive deployment, not only validation-set summaries. We instantiate Learn-then-Test for early-exit policies, prove distribution-free risk validity for certified selectors, characterize the compute price of searching a pre-ordered chain, and add joint risk-compute and estimated-weight shift certificates with explicit limitations. The controlled simulation verifies the central finite-sample behavior and the failure mode of naive threshold tuning with oracle risks, and the CIFAR-100 cache diagnostic shows the same qualitative safety/naive-failure pattern on a real early-exit model while exposing an important feasibility limitation. The remaining empirical burden is narrower but still real: the same violation-frequency design should be run on stronger vision backbones, language cascades, tabular cascades, and real shift settings before the draft claims broad real-model coverage.

---

## 14. Reproducibility checklist (TMLR)

For TMLR review, the submission will use anonymized supplementary material. The current reference implementation covers the p-values, certification procedures, selectors, controlled-simulation scripts, CIFAR cache adapter, and cache-based real-data evaluator; the reported synthetic and CIFAR diagnostic results are stored as JSON under `results/`; all $(\alpha,\delta,B)$, grids, seeds, and split counts used for reported experiments are specified or pre-registered, with exploratory CIFAR cells labeled separately; p-value implementations are unit-tested against their defining inequalities; violation-frequency experiments are scripted end-to-end; compute for the CIFAR diagnostic is an analytical MAC estimate at fixed input resolution; and no experiment tunes on test splits.

---

## Appendix A — Proofs

Throughout, losses are in $[0,1]$, calibration data are $n$ i.i.d. draws from $P$, and $\epsilon_n(\beta):=\sqrt{\ln(1/\beta)/(2n)}$. A p-value $p$ for a null $H$ is *valid* (super-uniform) if $\Pr_H(p\le u)\le u$ for all $u\in(0,1)$.

### A.1 Validity of the Hoeffding p-value (Eq. 2)

*Claim.* $p_\tau=\exp(-2n(\alpha-\hat R(\tau))_+^2)$ is valid for $H_\tau:R(\tau)\ge\alpha$.

*Proof.* Fix $\tau$ and write $R=R(\tau)$, $\hat R=\hat R(\tau)$. For $u\in(0,1)$, $p_\tau\le u \iff (\alpha-\hat R)_+\ge\epsilon_n(u) \iff \hat R\le\alpha-\epsilon_n(u)$ (the event is empty if $\alpha-\epsilon_n(u)<0$, making the probability $0\le u$ trivially). Under $H_\tau$, $R\ge\alpha$, so
$$
\Pr_{H_\tau}(p_\tau\le u)=\Pr(\hat R\le\alpha-\epsilon_n(u))\le\Pr(\hat R\le R-\epsilon_n(u))\le e^{-2n\epsilon_n(u)^2}=u,
$$
where the second inequality uses $\alpha\le R$ and the third is the lower-tail Hoeffding inequality for the mean of $n$ i.i.d. $[0,1]$ variables. Hence $p_\tau$ is super-uniform. The supremum over the composite null is attained at the boundary $R=\alpha$, as used above. $\square$

### A.2 Bonferroni FWER (Theorem 1)

*Claim.* With $\hat\Lambda_{\mathrm{Bonf}}=\{\tau:p_\tau\le\delta/|\Lambda|\}$ and valid $\{p_\tau\}$, $\Pr(\exists\tau\in\hat\Lambda_{\mathrm{Bonf}}:R(\tau)>\alpha)\le\delta$.

*Proof.* Let $\mathcal N=\{\tau:R(\tau)>\alpha\}$ be the true nulls. The bad event is $\bigcup_{\tau\in\mathcal N}\{p_\tau\le\delta/|\Lambda|\}$. By validity, $\Pr(p_\tau\le\delta/|\Lambda|)\le\delta/|\Lambda|$ for each $\tau\in\mathcal N$, so by the union bound the probability is at most $|\mathcal N|\cdot\delta/|\Lambda|\le|\Lambda|\cdot\delta/|\Lambda|=\delta$. Restricting any selection rule to the complement of the bad event yields a safe output, giving (P1); the infeasible case (no $\tau$ certified) deploys $\tau=\mathbf 1$ and reports infeasibility. $\square$

### A.3 Fixed-sequence FWER (Theorem 2a)

*Claim.* Testing a pre-specified order $\tau^{(K)},\dots,\tau^{(1)}$, certifying while $p_{\tau^{(k)}}\le\delta$ and stopping at the first failure, controls the probability of certifying any unsafe configuration at $\delta$, independent of $K$.

*Proof.* Let $j^\star$ be the position of the first true null in the testing order (the first tested $\tau^{(k)}$ with $R(\tau^{(k)})>\alpha$); if there is none, no unsafe config exists and the error probability is $0$. The procedure certifies a configuration only if it certifies everything earlier in the order, so it certifies an unsafe configuration only if it certifies the one at position $j^\star$ — which requires $p$ at that position $\le\delta$. Because the order is fixed *before* seeing the data, that hypothesis is a fixed true null, and by validity $\Pr(p\le\delta)\le\delta$. Thus the probability of certifying any unsafe configuration is at most $\delta$, with no dependence on $K$ (no correction is applied). $\square$

### A.4 Two-sided cost bound (Theorem 2b)

Stated and proved in full in the dedicated section *Appendix A.4* immediately following this appendix (kept separate for length). It establishes $\Pr(k_0\le k^\star\le k_1)\ge1-2\delta$ with band $\Delta_n=\epsilon_n(\delta)+\epsilon_n(\delta/K)$, so validity carries no $K$-penalty while the search cost is an $O(\sqrt{\ln(K/\delta)/n})$ efficiency band.

### A.5 Joint risk–compute control (Theorem 3)

*Claim.* With $\delta=\delta_R+\delta_C$, risk certified at $\delta_R$ (§A.2–A.4) and the uniform compute bound $U_C(\tau;\delta_C/K)=\hat C(\tau)+C_{\max}\sqrt{\ln(K/\delta_C)/(2n)}$ over the $K$ chain points, the dual selector satisfies $\Pr(R(\hat\tau)\le\alpha\ \wedge\ C(\hat\tau)\le B)\ge1-\delta$.

*Proof.* Define $\mathcal A=\{\forall\tau\in\hat\Lambda:R(\tau)\le\alpha\}$ (the risk-certification event) and $\mathcal B=\{\forall k:C(\tau^{(k)})\le U_C(\tau^{(k)};\delta_C/K)\}$. For $\mathcal B$: $c_{E_\tau(X)}\in[0,C_{\max}]$, so for each fixed $k$, upper-tail Hoeffding gives $\Pr(C(\tau^{(k)})>U_C(\tau^{(k)};\delta_C/K))\le\delta_C/K$; a union bound over the $K$ (data-independent) points gives $\Pr(\mathcal B^c)\le\delta_C$. By construction $\Pr(\mathcal A^c)\le\delta_R$. On $\mathcal A\cap\mathcal B$, the selected $\hat\tau\in\hat\Lambda$ has $R(\hat\tau)\le\alpha$, and $\hat\tau\in\hat\Lambda_B$ forces $C(\hat\tau)\le U_C(\hat\tau;\delta_C/K)\le B$. Boole: $\Pr((\mathcal A\cap\mathcal B)^c)\le\delta_R+\delta_C=\delta$. The denominator $K$ is deterministic, so no data-dependent correction is involved. $\square$

### A.6 Estimated-weight shift certificate (Theorem 4)

*Claim.* Under (⋆) $\mathbb E_P|\hat w-w|\le\eta$ (conditionally on $\mathcal D_1$), $0\le\hat w\le\hat W$, $\eta<\alpha$, and $Y\mid X$ invariant, the p-value (3) is valid for $H^Q_\tau:R_Q(\tau)>\alpha$, and certification at level $\delta$ gives $\Pr(R_Q(\hat\tau)\le\alpha)\ge1-\delta$ under $Q$.

*Proof.* Condition on $\mathcal D_1$. Then $\hat w$ is a fixed measurable function, and for $i\in\mathcal D_2$ the variables $\hat w(X_i)\ell_i$ are i.i.d. on $[0,\hat W]$ with common mean $\mu:=\mathbb E_P[\hat w\,\ell_{\mathrm{loss}}]$. Decompose $\mu=R_Q(\tau)+\mathbb E_P[(\hat w-w)\ell_{\mathrm{loss}}]$, using $R_Q(\tau)=\mathbb E_P[w\,\ell_{\mathrm{loss}}]$ (which holds because $Y\mid X$ is invariant). Since $\ell_{\mathrm{loss}}\in[0,1]$, the bias term is bounded: $|\mathbb E_P[(\hat w-w)\ell_{\mathrm{loss}}]|\le\mathbb E_P|\hat w-w|\le\eta$. Under $H^Q_\tau$, $R_Q(\tau)>\alpha$, hence $\mu>\alpha-\eta$. For observed $\hat R_Q=r\le\alpha-\eta$, the range-$\hat W$ Hoeffding inequality gives
$$
\Pr(\hat R_Q\le r\mid\mathcal D_1)\le\exp\!\Big(-\tfrac{2n_2}{\hat W^2}(\mu-r)^2\Big)\le\exp\!\Big(-\tfrac{2n_2}{\hat W^2}((\alpha-\eta)-r)^2\Big)=p^Q_\tau.
$$
So $p^Q_\tau$ is super-uniform under $H^Q_\tau$ conditionally on $\mathcal D_1$; since this holds for every realization of $\mathcal D_1$, it holds marginally (integrate over $\mathcal D_1$). The FWER arguments of A.2–A.4 then run conditionally on $\mathcal D_1$ and marginalize, giving $1-\delta$ validity under $Q$. The empirical-Bernstein variant replaces the range-$\hat W$ Hoeffding step with A.7 applied on $[0,\hat W]$, unchanged otherwise. $\square$

### A.7 Hoeffding–Bentkus and empirical-Bernstein p-values

**Hoeffding–Bentkus.** For $H:R\ge\alpha$ with $\hat R$ the mean of $n$ i.i.d. $[0,1]$ losses, the p-value $p^{\mathrm{HB}}=\min\{\exp(-n\,\mathrm{kl}(\hat R,\alpha)),\ e\,\Pr(\mathrm{Bin}(n,\alpha)\le\lceil n\hat R\rceil)\}$ (with $\mathrm{kl}$ the binary KL divergence, and $p^{\mathrm{HB}}=1$ if $\hat R\ge\alpha$) is valid; the first term is the Cramér–Chernoff (relative-entropy) bound and the second the Bentkus inequality, each super-uniform under $H$, so their minimum is super-uniform. We use this as the default in experiments and cite Bates et al. / Angelopoulos et al. for the two component inequalities.

**Empirical Bernstein (as implemented).** Let $\hat V_n$ be the unbiased sample variance of the $n$ losses (range $\rho$; $\rho=1$ unweighted, $\rho=\hat W$ weighted). The one-sided Maurer–Pontil inequality states that, for every $\beta\in(0,1)$,
$$
\Pr\Big(R-\hat R\ \ge\ \sqrt{\tfrac{2\hat V_n\ln(1/\beta)}{n}}+\tfrac{7\rho\ln(1/\beta)}{3(n-1)}\Big)\ \le\ \beta. \tag{MP}
$$
Define, for observed $(\hat R,\hat V_n)$ with $\hat R<\alpha$, the value $u^\star\ge0$ solving
$$
\alpha-\hat R\ =\ \sqrt{\tfrac{2\hat V_n}{n}}\,\sqrt{u^\star}\ +\ \tfrac{7\rho}{3(n-1)}\,u^\star
$$
(the right side is continuous and strictly increasing from $0$ to $\infty$ in $u^\star$, so the root is unique; in closed form, with $a=\sqrt{2\hat V_n/n}$, $b=7\rho/(3(n-1))$, $c=\alpha-\hat R$, $\sqrt{u^\star}=(-a+\sqrt{a^2+4bc})/(2b)$). Set $p^{\mathrm{EB}}=\exp(-u^\star)$ (and $p^{\mathrm{EB}}=1$ if $\hat R\ge\alpha$).

*Claim.* $p^{\mathrm{EB}}$ is valid for $H:R\ge\alpha$.

*Proof.* Because the deviation $D(\beta):=\sqrt{2\hat V_n\ln(1/\beta)/n}+7\rho\ln(1/\beta)/(3(n-1))$ is strictly increasing in $\ln(1/\beta)$, the definition of $u^\star$ gives the equivalence $p^{\mathrm{EB}}\le\beta\iff u^\star\ge\ln(1/\beta)\iff \alpha-\hat R\ge D(\beta)$. Under $H$, $R\ge\alpha$, so $\{\alpha-\hat R\ge D(\beta)\}\subseteq\{R-\hat R\ge D(\beta)\}$, and by (MP),
$$
\Pr_H(p^{\mathrm{EB}}\le\beta)=\Pr_H(\alpha-\hat R\ge D(\beta))\le\Pr_H(R-\hat R\ge D(\beta))\le\beta.
$$
Hence $p^{\mathrm{EB}}$ is super-uniform. Validity holds with $\hat V_n$ random because (MP) is itself an empirical-variance inequality, valid simultaneously over the realized $\hat V_n$. $\square$

(The reference implementation computes exactly this $u^\star$; the unit tests confirm $\Pr(p\le\beta)\le\beta$ under $R=\alpha$ by Monte Carlo for all three p-values.)

## Appendix A.4 — Full proof of the two-sided cost bound (Theorem 2b)

**Standing notation.**
(N1) Calibration data $\mathcal D=\{(X_i,Y_i)\}_{i=1}^n$ i.i.d. from $P$.
(N2) Chain $\tau^{(1)},\dots,\tau^{(K)}$ with deterministically nondecreasing compute $C_1\le\dots\le C_K$, where $C_k:=C(\tau^{(k)})$ (§4.1: raising the global threshold can only push each input to a deeper exit, so the per-input cost $c_{E_{\tau^{(k)}}(\cdot)}$ is pointwise nondecreasing in $k$, hence so is its expectation).
(N3) Per-point losses $\ell_{ik}:=\ell_{\mathrm{loss}}(\hat Y_{\tau^{(k)}}(X_i),Y_i)\in[0,1]$; $\hat R_k:=\frac1n\sum_i\ell_{ik}$, $R_k:=\mathbb E[\hat R_k]=R(\tau^{(k)})$.
(N4) Slack $\epsilon_n(\beta):=\sqrt{\ln(1/\beta)/(2n)}$. With the Hoeffding p-value $p_k=\exp(-2n(\alpha-\hat R_k)_+^2)$, the certification test $p_k\le\delta$ is **exactly** $\hat R_k\le\alpha-\epsilon_n(\delta)$.

The fixed-sequence walk from expensive to cheap certifies a suffix of the compute order; equivalently
$$
k^\star\;=\;\min\Big\{\,k:\ \hat R_j\le\alpha-\epsilon_n(\delta)\ \text{for all}\ j\ge k\,\Big\},
$$
with the convention $k^\star:=K+1$ (report infeasible, deploy $\tau^{(K)}$) if the set is empty. Output $\hat\tau=\tau^{(k^\star)}$. Let $k_0:=\min\{k:R_k\le\alpha\}$ be the cheapest truly-safe index ($k_0:=K+1$ if no config is safe).

**Lemma A.4.1 (Hoeffding, both tails).** For each fixed $k$ and any $t\ge0$,
$$
\Pr(\hat R_k\le R_k-t)\le e^{-2nt^2},\qquad \Pr(\hat R_k\ge R_k+t)\le e^{-2nt^2}.
$$
*Proof.* The $\{\ell_{ik}\}_{i=1}^n$ are i.i.d. in $[0,1]$ with mean $R_k$; apply Hoeffding's inequality to each tail. $\square$

*No independence across configurations $k$ is used anywhere below: every aggregation over $k$ is a union (Boole) bound, which holds for arbitrarily dependent events.*

**Input — validity (Appendix A.3, restated).** Because the testing order is fixed a priori and the walk stops at the first non-rejection, the event $\mathcal U:=\{k^\star\le K\ \text{and}\ R_{k^\star}>\alpha\}$ — "certify and output an unsafe config" — satisfies $\Pr(\mathcal U)\le\delta$. Write $\mathcal V:=\mathcal U^{c}$, so $\Pr(\mathcal V)\ge1-\delta$, and on $\mathcal V$ the method either reports infeasible or outputs a config with $R_{k^\star}\le\alpha$.

**Proposition A.4.2 (safety floor — lower bound).**
$$
\Pr\Big(C(\hat\tau)\ \ge\ \min_{k:\,R_k\le\alpha}C_k\Big)\ \ge\ 1-\delta .
$$
Under Assumption M ($R_1\ge\dots\ge R_K$) this is equivalently $\Pr(k^\star\ge k_0)\ge1-\delta$.
*Proof.* Work on $\mathcal V$. If the method certifies ($k^\star\le K$), then $R_{k^\star}\le\alpha$, so $\tau^{(k^\star)}$ is safe and $C(\hat\tau)=C_{k^\star}\ge\min_{k:R_k\le\alpha}C_k$ by definition of the minimum; if it reports infeasible, $\hat\tau=\tau^{(K)}$ and $C(\hat\tau)=C_K\ge\min_{k:R_k\le\alpha}C_k$ since $C_K$ is the largest compute. Either way the inequality holds on $\mathcal V$. Under Assumption M the safe set $\{k:R_k\le\alpha\}=\{k_0,\dots,K\}$ is a suffix and $C_k$ is nondecreasing, so $\min_{k\ge k_0}C_k=C_{k_0}$; moreover any $k<k_0$ has $R_k\ge R_{k_0-1}>\alpha$, so a safe output forces $k^\star\ge k_0$. $\square$

**Proposition A.4.3 (efficiency ceiling — upper bound).** For any $\beta\in(0,1)$ put
$$
\Delta_n(\beta):=\epsilon_n(\delta)+\epsilon_n(\beta/K),\qquad
k_1(\beta):=\min\{k:\ R_j\le\alpha-\Delta_n(\beta)\ \text{for all}\ j\ge k\}.
$$
Then $\Pr\big(k^\star\le k_1(\beta)\big)\ge1-\beta$.
*Proof.* If every $j\ge k_1(\beta)$ certifies — i.e. $\hat R_j\le\alpha-\epsilon_n(\delta)$ — then by definition $k^\star\le k_1(\beta)$. Hence
$$
\Pr\big(k^\star> k_1(\beta)\big)\le\Pr\Big(\exists\,j\ge k_1(\beta):\ \hat R_j>\alpha-\epsilon_n(\delta)\Big).
$$
Fix $j\ge k_1(\beta)$. By the defining property of $k_1(\beta)$, $R_j\le\alpha-\Delta_n(\beta)=\alpha-\epsilon_n(\delta)-\epsilon_n(\beta/K)$, so $\alpha-\epsilon_n(\delta)\ge R_j+\epsilon_n(\beta/K)$, and the upper tail of Lemma A.4.1 with $t=\epsilon_n(\beta/K)$ gives
$$
\Pr\big(\hat R_j>\alpha-\epsilon_n(\delta)\big)\le\Pr\big(\hat R_j>R_j+\epsilon_n(\beta/K)\big)\le e^{-2n\,\epsilon_n(\beta/K)^2}=\beta/K .
$$
A union bound over the at most $K$ indices $j\in\{k_1(\beta),\dots,K\}$ gives $\Pr(k^\star> k_1(\beta))\le K\cdot(\beta/K)=\beta$. $\square$

(Proposition A.4.3 uses neither Assumption M nor cross-config independence.)

**Theorem A.4.4 (two-sided cost bound).** With $\beta=\delta$, write
$$
\Delta_n:=\epsilon_n(\delta)+\epsilon_n(\delta/K)=O\!\Big(\sqrt{\tfrac{\ln(K/\delta)}{n}}\Big),\qquad k_1:=k_1(\delta).
$$
Then on an event of probability at least $1-2\delta$,
$$
k_0\ \le\ k^\star\ \le\ k_1,\qquad\text{and}\qquad 0\ \le\ C(\hat\tau)-C_{k_0}\ \le\ C_{k_1}-C_{k_0}.
$$
Under Assumption M the thresholds collapse to $k_0=\min\{k:R_k\le\alpha\}$ and $k_1=\min\{k:R_k\le\alpha-\Delta_n\}$ (the "for all $j\ge k$" conditions reduce to single-index conditions because $R_k$ is nonincreasing), so the excess compute $C_{k_1}-C_{k_0}$ is exactly the compute spanned by configurations whose true risk lies in the boundary band $(\alpha-\Delta_n,\ \alpha]$.
*Proof.* Intersect the events of Propositions A.4.2 and A.4.3 (with $\beta=\delta$); by Boole's inequality the intersection has probability $\ge1-2\delta$. There $k_0\le k^\star\le k_1$, and the compute inequalities follow because $C_k$ is nondecreasing in $k$. The Assumption-M simplification is immediate from monotone $R_k$. $\square$

**Corollary A.4.5 (consistency and rate).** Suppose along the band the chain relates compute to risk Lipschitz-ly: $\exists\,\Lambda_C<\infty$ with $C_{k_1}-C_{k_0}\le\Lambda_C\,(R_{k_0}-R_{k_1})\le\Lambda_C\,\Delta_n$. Then with probability $\ge1-2\delta$,
$$
C(\hat\tau)-\min_{k:\,R_k\le\alpha}C_k\ \le\ \Lambda_C\,\Delta_n\ =\ O\!\Big(\sqrt{\tfrac{\ln(K/\delta)}{n}}\Big)\ \xrightarrow[n\to\infty]{}\ 0 .
$$
*Proof.* Immediate from Theorem A.4.4 and the assumed Lipschitz relation. $\square$

**Remarks.**
*(i) Where each ingredient enters.* Validity (Prop. A.4.2) is $K$-free: it is the level-$\delta$ fixed-sequence guarantee. The $\ln K$ enters only the efficiency band $\Delta_n$ (Prop. A.4.3), through the union bound over the certified suffix. This is the precise sense of "the cost of a length-$K$ search is paid in efficiency, not validity."
*(ii) Decoupled confidences.* Validity holds at level $\delta$ and the ceiling at an independent level $\beta$; reporting the sandwich at $1-\delta-\beta$ lets one buy a tighter efficiency statement (smaller $\beta$, wider band) without touching validity.
*(iii) Role of Assumption M.* Used only to turn the safety floor into the index form $k^\star\ge k_0$ and to collapse $k_0,k_1$ into clean risk thresholds. The probabilistic content of Props. A.4.2–A.4.3 is assumption-free; when M fails, use the Holm fallback (validity preserved) and read the lower bound in compute form $C(\hat\tau)\ge\min_{\text{safe}}C_k$. A graphical procedure could be substituted in future work, but it is not evaluated here.
*(iv) Tighter p-values.* Replacing Hoeffding by Hoeffding–Bentkus or empirical-Bernstein shrinks $\epsilon_n(\delta)$, hence $\Delta_n$, with no change to the argument.

---

## Appendix B — Extended experimental detail

The controlled-simulation configuration of §10.1 (generative parameters, chain grid, $(\alpha,\delta)$ values, split counts, and seeds) is specified in the reference implementation and reproduced by its `experiments/` scripts. The generated artifacts are `results/e1_validity.json`, `results/e2_chain_vs_bonferroni.json`, `results/e5_joint_control.json`, and `results/e6_shift_eta_sweep.json`.

The CIFAR-100 diagnostic in §10.2 uses `adapters/cifar_branchy.py` to train/cache a Branchy ResNet-56-style model with exits after the stem, stage 1, stage 2, and stage 3. The cache artifact is `cache/cifar_branchy_resnet56.npz`; the pre-registered and exploratory cache-evaluation artifacts are `results/cifar_branchy_real_cache_eval.json` and `results/cifar_branchy_real_cache_eval_exploratory.json`. The cache contains $n=10000$ test examples, $K=100$ scalar thresholds, top-softmax confidence, 0–1 loss, and cumulative analytical MAC estimates $[443{,}968,\;42{,}911{,}296,\;84{,}331{,}648,\;125{,}753{,}600]$. The checkpoint records epoch 80 and per-exit test accuracies $[0.0646,\;0.3040,\;0.6147,\;0.6945]$. Remaining ImageNet, language-cascade, tabular, corruption-suite, and weight-estimation details will be tabulated when those studies are run.
