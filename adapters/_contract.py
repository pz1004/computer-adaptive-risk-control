"""
Per-exit cache contract for CARC adapters.

An adapter writes an `.npz` with:
  scores      : (n, L) confidence at each exit, higher means more willing to exit
  correct     : (n, L) per-exit correctness indicators
  loss        : (n, L) per-exit bounded losses in [0, 1]
  exit_cost   : (L,) cumulative cost to reach each exit
  thresholds  : (K,) chain thresholds, increasing compute when raised
  loss_matrix : (n, K) chain losses from `carc.chain.build_chain`
  cost_matrix : (n, K) chain costs from `carc.chain.build_chain`
  meta_json   : JSON string with dataset/model/split/provenance metadata
"""
