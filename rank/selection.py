"""Choose a re-ranking configuration on the validation split, never on test."""


def select_on_validation(val_summary, reference, delta=0.05, objective="gini_top1", accuracy="hit3"):
    """Among configs whose validation accuracy is within `delta` (relative) of the
    reference config (the reference included), return the one minimising
    `objective`; ties go to the higher accuracy."""
    ref = val_summary.loc[val_summary.config == reference, accuracy].iloc[0]
    adm = val_summary[val_summary[accuracy] >= (1 - delta) * ref]   # the reference itself is admissible
    if adm.empty:
        return reference
    return adm.sort_values([objective, accuracy], ascending=[True, False]).config.iloc[0]
