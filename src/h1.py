"""H1: do most errors come from overlapping intent definitions rather than weak models? Never reads test.csv.

Pre-registered parts (PREREGISTRATION.md, including the 2026-11-09 amendment):
  (a) >= 50% of the main model's OOF errors fall within confusable intent groups.
      Edge if two intents are confused >= 5 times in total; groups are connected components.
  (b) In about 50 hand-labeled errors, the most common cause is "ambiguous_label" or
      "vague_message", not "vocabulary_gap".
  (c) The top-10 confused pairs are shared across model families: >= 5 of 10 in common
      between LR and KNN, and between LR and the random forest.
Also reported: accuracy after merging each confusable group, top-3 accuracy, and coarse-category
accuracy (only once the author has approved reports/coarse_groups.json).

Usage: python -m src.h1   (writes reports/h1.json; creates reports/error_labels.csv for labeling
                           if it doesn't exist yet, and never overwrites existing labels)
"""
import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.config import DATA_PROCESSED, REPORTS, SEED
from src.metrics import accuracy, macro_f1

EDGE_THRESHOLD = 5
EDGE_SENSITIVITY = [3, 8, 10]
FAMILIES = {"LR": "M1_L2", "KNN": "M2_KNN", "RF": "M4_RF_SVD"}
N_LABEL = 50
CAUSES = ["ambiguous_label", "vague_message", "vocabulary_gap", "possible_mislabel"]
LABELS_CSV = REPORTS / "error_labels.csv"
COARSE_JSON = REPORTS / "coarse_groups.json"

COARSE_DRAFT = {
    "cards": ["activate_my_card", "apple_pay_or_google_pay", "card_about_to_expire", "card_acceptance", "card_arrival",
              "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
              "card_payment_not_recognised", "card_swallowed", "contactless_not_working", "declined_card_payment",
              "disposable_card_limits", "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
              "getting_virtual_card", "order_physical_card", "pending_card_payment", "reverted_card_payment?",
              "supported_cards_and_currencies", "virtual_card_not_working", "visa_or_mastercard"],
    "transfers": ["balance_not_updated_after_bank_transfer", "beneficiary_not_allowed", "cancel_transfer",
                  "declined_transfer", "direct_debit_payment_not_recognised", "failed_transfer", "pending_transfer",
                  "receiving_money", "transfer_fee_charged", "transfer_into_account",
                  "transfer_not_received_by_recipient", "transfer_timing"],
    "top_ups": ["automatic_top_up", "balance_not_updated_after_cheque_or_cash_deposit", "pending_top_up",
                "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
                "top_up_limits", "top_up_reverted", "topping_up_by_card", "verify_top_up"],
    "exchange": ["card_payment_wrong_exchange_rate", "exchange_charge", "exchange_rate", "exchange_via_app",
                 "fiat_currency_support", "wrong_exchange_rate_for_cash_withdrawal"],
    "identity_security": ["change_pin", "compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone",
                          "passcode_forgotten", "pin_blocked", "unable_to_verify_identity", "verify_my_identity",
                          "verify_source_of_funds", "why_verify_identity"],
    "other": ["Refund_not_showing_up", "age_limit", "atm_support", "cash_withdrawal_charge",
              "cash_withdrawal_not_recognised", "country_support", "declined_cash_withdrawal",
              "edit_personal_details", "extra_charge_on_statement", "pending_cash_withdrawal", "request_refund",
              "terminate_account", "transaction_charged_twice", "wrong_amount_of_cash_received"],
}


def load(name):
    d = np.load(REPORTS / "oof" / f"{name}.npz", allow_pickle=True)
    return d["proba"].astype(float), d["classes"]


def pair_counts(y, pred, classes):
    """Symmetric matrix: S[i, j] = times i was predicted as j plus times j was predicted as i."""
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    M = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred):
        if t != p:
            M[idx[t], idx[p]] += 1
    return M + M.T


def confusable_groups(S, classes, threshold):
    """Connected components of the graph with an edge wherever S >= threshold (size >= 2 only)."""
    rows, cols = np.nonzero(np.triu(S >= threshold, 1))
    n_comp, labels = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=S.shape), directed=False)
    groups = [sorted(classes[labels == g]) for g in range(n_comp) if (labels == g).sum() >= 2]
    return sorted(groups, key=len, reverse=True)


def share_within_groups(y, pred, groups):
    member = {c: g for g, grp in enumerate(groups) for c in grp}
    wrong = y != pred
    inside = [member.get(t, -1) == member.get(p, -2) for t, p in zip(y[wrong], pred[wrong])]
    return float(np.mean(inside)), int(np.sum(inside)), int(wrong.sum())


def chance_share(y, pred, groups, n_classes):
    """Share expected inside groups if each error went to a uniformly random *wrong* intent.

    For an error on true intent t in a group of size g, a random wrong label lands in the
    same group with probability (g - 1) / (n_classes - 1). Big groups raise this baseline,
    so it shows how much of (a) is just group size.
    """
    size = {c: len(grp) for grp in groups for c in grp}
    return float(np.mean([(size.get(t, 1) - 1) / (n_classes - 1) for t in y[y != pred]]))


def merge_labels(labels, groups):
    mapping = {c: "+".join(grp) for grp in groups for c in grp}
    return np.array([mapping.get(c, c) for c in labels], dtype=object)


def top_pairs(y, pred, classes, n=10):
    S = pair_counts(y, pred, classes)
    i, j = np.triu_indices(len(classes), 1)
    order = np.argsort(-S[i, j], kind="stable")[:n]
    return [(classes[i[o]], classes[j[o]], int(S[i[o], j[o]])) for o in order]


def write_labeling_sheet(df, y, pred, proba, classes):
    """A random sample of N_LABEL main-model errors for the author to label. Never overwrites."""
    if LABELS_CSV.exists():
        return False
    wrong_idx = np.flatnonzero(y != pred)
    pick = np.sort(np.random.default_rng(SEED).choice(wrong_idx, N_LABEL, replace=False))
    order = np.argsort(-proba, axis=1)
    col = {c: i for i, c in enumerate(classes)}
    rows = [{"row": int(r), "text": df["text"][r], "true_intent": y[r], "predicted_intent": pred[r],
             "p_predicted": round(float(proba[r, order[r, 0]]), 3), "p_true": round(float(proba[r, col[y[r]]]), 3),
             "second_guess": classes[order[r, 1]], "cause": "", "notes": ""} for r in pick]
    pd.DataFrame(rows).to_csv(LABELS_CSV, index=False)
    return True


def labeled_causes():
    if not LABELS_CSV.exists():
        return None
    causes = pd.read_csv(LABELS_CSV, dtype=str)["cause"].fillna("").str.strip()
    if (causes == "").any():
        return {"status": "pending", "labeled": int((causes != "").sum()), "of": len(causes)}
    bad = set(causes) - set(CAUSES)
    if bad:
        raise ValueError(f"Unknown causes in {LABELS_CSV.name}: {bad}. Use one of {CAUSES}.")
    shares = causes.value_counts(normalize=True).reindex(CAUSES, fill_value=0.0)
    top = shares.idxmax()
    return {"status": "done", "n": len(causes), "shares": shares.round(3).to_dict(), "most_common": top,
            "supported": top in ("ambiguous_label", "vague_message")}


def main():
    df = pd.read_csv(DATA_PROCESSED / "train_split.csv")
    y = df["category"].to_numpy()
    proba, classes = load("M1_L2")
    pred = classes[proba.argmax(axis=1)]
    S = pair_counts(y, pred, classes)

    # (a) error share within confusable groups, at the pre-registered threshold and the sensitivity values.
    part_a = {}
    for t in [EDGE_THRESHOLD] + EDGE_SENSITIVITY:
        groups = confusable_groups(S, classes, t)
        share, inside, total = share_within_groups(y, pred, groups)
        part_a[str(t)] = {"share": share, "chance_share": chance_share(y, pred, groups, len(classes)),
                          "errors_inside": inside, "errors_total": total, "n_groups": len(groups),
                          "group_sizes": [len(g) for g in groups], "intents_in_groups": sum(len(g) for g in groups)}
    groups = confusable_groups(S, classes, EDGE_THRESHOLD)
    part_a["groups_at_5"] = groups
    part_a["supported"] = part_a[str(EDGE_THRESHOLD)]["share"] >= 0.5

    # Merging each confusable group into one label: how much does the score rise?
    y_m, pred_m = merge_labels(y, groups), merge_labels(pred, groups)
    merge = {"accuracy_before": accuracy(y, pred), "accuracy_after": accuracy(y_m, pred_m),
             "macro_f1_before": macro_f1(y, pred), "macro_f1_after": macro_f1(y_m, pred_m),
             "n_labels_after": int(len(set(y_m)))}
    top3 = float(np.mean([t in classes[o] for t, o in zip(y, np.argsort(-proba, axis=1)[:, :3])]))

    # (c) top-10 confused pairs shared across model families.
    pairs = {fam: top_pairs(y, load(m)[1][load(m)[0].argmax(axis=1)], classes) for fam, m in FAMILIES.items()}
    lr = {(a, b) for a, b, _ in pairs["LR"]}
    overlap = {fam: len(lr & {(a, b) for a, b, _ in pairs[fam]}) for fam in ("KNN", "RF")}
    part_c = {"top10": {f: [list(p) for p in v] for f, v in pairs.items()}, "shared_with_LR": overlap,
              "supported": all(v >= 5 for v in overlap.values())}

    # (b) hand-labeled causes.
    created = write_labeling_sheet(df, y, pred, proba, classes)
    part_b = labeled_causes()

    # Coarse categories: draft until the author approves.
    if not COARSE_JSON.exists():
        assert sorted(sum(COARSE_DRAFT.values(), [])) == sorted(classes), "draft must cover every intent exactly once"
        COARSE_JSON.write_text(json.dumps({"approved": False, "groups": COARSE_DRAFT}, indent=2))
    coarse_def = json.loads(COARSE_JSON.read_text())
    coarse = {"status": "awaiting author approval of reports/coarse_groups.json"}
    if coarse_def["approved"]:
        to_coarse = {c: g for g, members in coarse_def["groups"].items() for c in members}
        yc, pc = np.array([to_coarse[c] for c in y]), np.array([to_coarse[c] for c in pred])
        coarse = {"status": "done", "accuracy": accuracy(yc, pc), "share_of_errors_within_same_coarse_category":
                  float(np.mean((yc == pc)[y != pred]))}

    verdict = "pending (hand labels)" if part_b is None or part_b["status"] != "done" else \
        ("supported" if part_a["supported"] and part_b["supported"] and part_c["supported"] else "not supported")
    result = {"a_within_group_share": part_a, "b_hand_labels": part_b, "c_cross_model_pairs": part_c,
              "merge": merge, "top3_accuracy": top3, "coarse": coarse, "verdict": verdict,
              "labeling_sheet_created_now": created}
    (REPORTS / "h1.json").write_text(json.dumps(result, indent=2))

    for t in [EDGE_THRESHOLD] + EDGE_SENSITIVITY:
        a = part_a[str(t)]
        print(f"(a) edge>={t:2d}: {a['share']:.3f} of errors inside groups (chance {a['chance_share']:.3f})  "
              f"({a['n_groups']} groups, sizes {a['group_sizes']})")
    print(f"merge: accuracy {merge['accuracy_before']:.4f} -> {merge['accuracy_after']:.4f}; "
          f"macro-F1 {merge['macro_f1_before']:.4f} -> {merge['macro_f1_after']:.4f}  ({merge['n_labels_after']} labels)")
    print(f"top-3 accuracy {top3:.4f}")
    print(f"(c) shared with LR top-10: {overlap}  supported={part_c['supported']}")
    print(f"(b) {part_b}")
    print(f"coarse: {coarse['status']}")
    print(f"H1 verdict: {verdict}")


if __name__ == "__main__":
    main()
