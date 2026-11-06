# ---- Kaggle kernel: baseline ("before") eval of Tiny Aya Earth ----
install_base()
rc = eval_parallel(f"{W}/results/baseline")
log("baseline done", rc, "total s", round(time.time() - T0))
