# Rights, provenance and attribution

This release publishes author-supplied research code and numerical outputs for
review and saved-record recomputation. It does not relicense third-party work,
datasets, logos, images, fonts, or model weights, and does not assign a new blanket
software license where none previously existed.

- OSLD: Muhammet Bastan et al., *Large Scale Open-Set Deep Logo Detection*, 2019.
  The official dataset is CC BY-NC 4.0 for research; individual images/logos remain
  with their rights owners. No dataset image pixels or full annotation archive is
  redistributed. Numerical model outputs and the four annotated feature-grid
  examples are identified in BGER_WWW27_reproducibility/DATA_ACCESS.md.
- QMUL-OpenLogo: Hang Su, Xiatian Zhu and Shaogang Gong, *Open Logo Detection
  Challenge*, BMVC 2018. The official dataset is for academic research and image
  copyrights remain with original owners. Only protocol/evaluation metadata,
  derived target-equivalence evidence and numerical outputs are included.
  Official source: https://hangsu0730.github.io/qmul-openlogo/ .
- CUB-200-2011: Catherine Wah et al., Caltech technical report CNS-TR-2011-001.
  No bird images are redistributed. The supplied per-image records contain model
  correctness/hit arrays and identifiers, not source pixels.
- ABN-box follows the official attention-branch architecture of Hiroshi Fukui et
  al., CVPR 2019, with the attribution retained in `BGER_WWW27_reproducibility/logo_retrieval_osld/code/abn_box.py`. The upstream
  MIT notice (copyright 2017 Wei Yang) is reproduced verbatim in
  `THIRD_PARTY_LICENSES/ATTENTION_BRANCH_NETWORK_MIT.txt`.
- The multi-contour MaxBoxAccV2 geometry follows clovaai/wsolevaluation, commit
  `96f4b88679af41f005824661f47e6fe3628af2aa`. Its NAVER copyright/permission notice
  is preserved in `THIRD_PARTY_LICENSES/WSOL_EVALUATION_MIT.txt`.
- DINOv2, OpenCLIP/LAION, torchvision/ImageNet, Grad-CAM and similarity-score
  decomposition sources retain their existing citations and provenance. Official
  weight identifiers/digests are recorded for inspection; no model binaries are
  bundled and no license for externally obtained weights is expanded here.

The exact source notices and third-party names are attribution, not a claim of
manuscript authorship. Required third-party notices must not be anonymized away.
