# Pre-test amendment: OSLD-SingleLogo-Retrieval-v1

Recorded 2026-10-16, before any OSLD model output exists (no training, no
validation and no test predictions). Authority: the author's written instruction
of 2026-10-16 (`prompt/Claude_OSLD_Web_to_Submission_Prompt.md`, SHA256 in the
data receipt). This amendment **replaces** the DeepFashion In-shop design for the
Web study; it is not a retroactive approval of the In-shop protocol, and the
In-shop design is retired without having produced any result.

## What changed and why

| Item | In-shop design (retired) | OSLD-SingleLogo-Retrieval-v1 | Reason |
|---|---|---|---|
| Data | DeepFashion In-shop (access blocked) | OSLD official release (CC BY-NC 4.0, research only), downloaded from the official Dropbox link | author's data choice |
| Task | same-product image retrieval | product image → canonical-logo library retrieval of **unseen logo identities** | release structure |
| Relevance | same product ID | same logo identity (canonical file stem); all templates of the identity are relevant | release annotations |
| Query cohort | all query images | **annotated single-logo subset**: exactly one box, labeled (not `__unknown__`), valid coordinates, matched template decodable; exclusions counted by reason | no multi-object detection; no model-based filtering |
| Gallery | product gallery images | canonical logo images; dev = templates of train ∪ val identities; test = full decodable library (distractor identities included) | open-set matching |
| Development split | 15% of train identities | official `val` split (identity-disjoint) | release provides it |
| Training inputs | product images | train-cohort product images (`has_box=True`) + canonical images of train identities (`has_box=False`, never a full-image box, never all-background) | template-domain coverage |
| Box target | 224 mask → 7×7 average pool > 0.5 | 7×7 **any positive-area overlap** (floor lower / ceil upper, clipped) for every spatially supervised arm | small logos would otherwise give all-zero targets (measured below) |
| Arms | 4 (15 runs) | 4 + **ABN-box** (published-method adaptation): **19 runs** = 9 seed-0 development + 10 confirmation | closest published supervised-attention baseline |
| Localization endpoint | query and gallery boxes | **query only**; templates have no product-region ground truth (gallery maps kept descriptively) | release has no template boxes |
| Joint endpoint | J@1 = R·Lq·Lg | **Jq@1 = R·Lq** | same reason |
| Confirmatory family | H1–H3 (BGER−CE) | **H1-logo** R@1 > −0.01 and **H2-logo** query Loc > 0, Holm over two, α=.05 | gallery H3 retired |
| Cluster unit | query product | **query logo identity** | dependence among queries of one identity |
| Utility reference | none | uniform-random R@1 from label counts | absolute usefulness |
| A5 | proposed | adopted as descriptive diagnostic on fixed CE-seed-0 positive/negative templates; centre prior as candidate-independent reference | author instruction |

Unchanged: 224×224 input; ResNet-50 (torchvision ImageNet V1); 15 epochs;
final checkpoint; batch 64; SGD/Nesterov with the cosine schedule; FP32;
λ ∈ {0.5, 2} at seed 0 with the within-family development rule (R@1 within
1 pp of the arm's best, then highest development query localization, ties to the
smaller λ); thresholds locked on development; stable cosine primary readout
with the query's own norm derivative; forward decomposition as secondary;
10,000 shared bootstrap draws, basic intervals, centered one-sided p-values.

## Implementation changes made with this amendment (branch `e1_osld`)

- `code/osld_data.py` (scan, cohort, sets, loader, any-overlap target), `code/osld_release_check.py`, `code/make_osld_fixture.py`.
- `code/abn_box.py`: official ImageNet ResNet-50 ABN structure (commit 5e418783), box supervision on the attention logits; common-template initialization of shared tensors; locked-seed attention branch.
- Evaluation: query-only localization, Jq@1, random-retrieval reference; analysis: two-hypothesis family, five-arm contrasts, identity clusters.
- Trainer performance fix (applies to all arms): full checkpoints every 200 steps and at epoch end instead of every step; resume semantics unchanged. Step/data timing recorded.
- Guards: OSLD identities, receipt rules (public-license authorization evidence), B0 timing plan (30 batch-64 steps per arm on train; 150 dev queries; 40 fixed-pair readouts; official test never built).
