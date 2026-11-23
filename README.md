# Questioning Image Retrieval: answers and reproduction

Code and data for the CVPR 2027 submission *Questioning Image Retrieval: Cheap Verifiers Do More Than Rerank*. Every number in the paper is a function of the verifier's saved answers: the questions asked for each query,
the photos the search returned, and the answer to each question about each photo. The answers are released as a
Hugging Face dataset of parquet tables (102 MB, about 18 million answers):
[QuestioningRetrieval/questioning-retrieval-answers](https://huggingface.co/datasets/QuestioningRetrieval/questioning-retrieval-answers). Reproducing the paper needs no verifier call,
no image and no embedding.

```
1. answers   the saved answers, one folder of tables per source        python scripts/fetch_data.py
2. numbers   every number of the paper, recomputed from the answers    python paper/paper_numbers.py
3. tables    the supplement's tables and plots, from the answers      python paper/build.py
```

```
python -m pip install -r requirements.txt
python scripts/fetch_data.py          # 101 files, 102 MB, into ../questioning-retrieval-data, each checked by sha256
python paper/paper_numbers.py         # every number the answers determine, each set against the printed value (a few minutes, CPU)
```

The last line printed is `161 of 161 numbers printed in the paper match; ...` (165 with GPT-4o's answers, below). The data
folder is `$QR_DATA` if set, else the `data` entry of an untracked `local.json` (`{"data": "D:/somewhere"}`), else
`../questioning-retrieval-data` next to this folder.

## 1. The answers

One folder per source. Each holds the same four tables, joined on `query_id` and `image_id`:

| table | one row per | columns |
|---|---|---|
| `queries.parquet` | query | `source`, `benchmark`, `subset`, `split`, `query_id`, `query` (the text the verifier read), `relevant` (image ids of the right answers), `meta` (JSON; INQUIRE: supercategory, category, iconic group) |
| `questions.parquet` | question | `query_id`, `question_id` (`holistic`, `sub0`..`sub4`, `relevance`), `role`, `kind` (`noul`: yes or no; `score`: 1 to 5), `text`, `weight` (statements, 0.5 to 1.5), `levels` (the labels of the 1-to-5 scale) |
| `candidates.parquet` | photo checked | `query_id`, `image_id`, `search_rank`, `search_score` (the search's similarity; min-max scaled over the pool, it is $c$ in Eq. 2), `siglip2_score` (a second score where there is one, see below) |
| `answers.parquet` | photo and question | `query_id`, `image_id`, `question_id`, `p_yes`, `p_levels` (the five levels' probabilities), `expected` (the 1-to-5 answer's expected value scaled to [0, 1]), `verifier`, `input_tokens` (of the photo's request), `error` |

| source | benchmark, queries | photos per query | used in |
|---|---|---|---|
| `docci_s1` .. `docci_s4`, `docci_all` | DOCCI test, 500 descriptions: the first 1 to 4 sentences to arrive, and all of them | SigLIP2's top 100 | Tables 1-3, Figures 1, 2, 4, 5 |
| `docci_heldout` | DOCCI test, 500 other descriptions, the whole text as one query | SigLIP2's top 200 | Table 2 (held out) |
| `docci_deep200_s1` .. `s4` | the same 500 descriptions, looking further | the search's top 200, then the photos the answers chose (`steered.parquet`) | Sec. 6, Table 3 |
| `inquire_rerank_test`, `inquire_rerank_val` | INQUIRE-Rerank, 160 and 40 queries | INQUIRE's fixed 100 | Table 1, Sec. 4 |
| `inquire_full_vith_lc_test` / `_val` | INQUIRE-Fullrank, 200 and 50 queries | CLIP ViT-H/14's top 500 | Tables 1, 3, 4, Figure 3 |
| `inquire_full_siglip2_lc_test` / `_val` | INQUIRE-Fullrank, 200 and 50 queries | SigLIP2's top 500 | Table 1 |
| `clef_flash` | the open verifier's answers to the same questions; `source` and `query_key` say which run | 100 per query | Sec. 4, supplement |
| `winoground` | Winoground, 400 examples, one yes or no question per caption and image; `mode`: the run and its two controls | 4 per example | Table 1, Sec. 5 |
| `bank_inquire_rerank_test`, `bank_inquire_full_vith_lc_val`, `bank_docci_s1` | the larger question set (15.7 questions per photo), plainly and with the LLM's brief; four tables of their own (`queries`, `photos`, `questions`, `answers`), with the earlier run's seven answers and search scores on the same photos | 100, 500, 100 per query | Sec. 4, Table 4 |
| `bank_inquire_wide_test`, `bank_inquire_wide_val` | the larger question set, plainly and with the brief, on every photo checked for an INQUIRE-Fullrank query in four earlier runs; `photos` adds each photo's rank and score in those runs' pools | about 900 per query | Table 4 (uncertainty) |

Side tables: `docci/targets.parquet` (each DOCCI target's sentences and the random order in which they arrive),
`docci/images.parquet` (DOCCI image ids to DOCCI's own example ids), `inquire_full_*/firststage.parquet` (the pool
encoder's top 1,000, used to label the photos after the 500 checked), `docci_deep200_*/steered.parquet` (the 100
photos the answers chose), `inquire_taxa/taxa.parquet` (iNat24's category of every INQUIRE photo in the first stages,
for Sec. 7's species errors), `inquire_full_streams/streams.parquet` (every saved SigLIP2 search of an INQUIRE-Fullrank
query, for Table 4's feedback rows), `inquire_full_steer/searches.parquet` (Table 4's steered search: SigLIP2's search
for each query, and the same steered by the answers or by the labels once 100 or 500 photos are checked, top 1,000 each),
`costs/ledgers.parquet` (requests, tokens and prices of the INQUIRE-Rerank test run) and
`inquire_gpt4o_pool/pool.parquet` (the photos GPT-4o reranks in the INQUIRE paper: CLIP ViT-H's top 100 for the query
as written, with their iNat24 paths).

Search scores. INQUIRE-Rerank test and DOCCI: SigLIP2 so400m/16-384 with the query lowercased (`siglip2_score` keeps the
score of the query as written on INQUIRE-Rerank test). INQUIRE-Rerank validation kept the query as written.
INQUIRE-Fullrank: the encoder that built the pool, with its own rank and score; the ViT-H pool keeps SigLIP2's score as
`siglip2_score`. DOCCI with sentences: one SigLIP2 vector per sentence, averaged.

Image ids are the benchmarks' own: INQUIRE's image id (iNaturalist 2024), DOCCI `i<row>` of the test split
(`docci/images.parquet` gives DOCCI's example id), Winoground's example id with caption and image 0 or 1. No image is
released; the figures read them from each benchmark's own release.

## 2. Every number from the answers

`qr/score.py` is the scoring rule of Sec. 3: the weighted geometric mean of the statements (Eq. 1) and the final score
(Eq. 2), the lead, and the ablations (other weights, the search score left out, fusion without weights).
`qr/metrics.py` holds INQUIRE's AP, the threshold of Sec. 5 and the protocols around it (40 random halves of the
targets, used both ways; paired bootstrap over queries). `qr/bank.py` scores the larger question set as the analysis
behind Sec. 4 did, `qr/uncertainty.py` reads its "cannot tell" answers for Table 4, and `qr/feedback.py` Table 4's
feedback to the search. `paper/paper_numbers.py` recomputes the paper's numbers in groups:

| group | paper | sources |
|---|---|---|
| `table1` | Table 1 | `inquire_rerank_test`, `inquire_full_*_test`, `docci_all`, `winoground` |
| `sec4` | Sec. 3-4 text: lowercasing, the whole query alone, pool recall, depth, cost, the larger question set, the paired gain over GPT-4o | `inquire_rerank_test`, `inquire_full_*_test`, `bank_*`, `inquire_gpt4o_pool` (+ GPT-4o's answers) |
| `depth` | supplement: which answers make depth pay, with confidence intervals | `inquire_full_*_test` |
| `table2` | Table 2 and the threshold | `docci_s1`..`docci_all`, `docci_heldout` |
| `sec5` | Sec. 5 text: AUROC of the lead, the lead with fewer questions, vague queries, Winoground | `docci_*`, `winoground` |
| `table3` | Table 3 (fixed budgets, the adaptive policies, expand before asking), the loop's breakdown, the next-batch predictor, the match probability's calibration | `docci_*`, `docci_deep200_*`, `docci/targets`, `inquire_full_vith_lc_*` |
| `sec6` | Sec. 6 text: candidate expansion | `docci_deep200_*` + `steered` |
| `sec7` | Sec. 7 and Table 4: missing targets, perfect verification, the best single question, species errors, uncertainty, rewording, LLM text and the steered search | `docci_*`, `inquire_full_*`, `inquire_taxa`, `bank_inquire_wide_*`, `inquire_full_streams`, `inquire_full_steer` |
| `figures` | the numbers printed in Figures 2 and 3 | `docci_s2`, `inquire_full_vith_lc_test` |
| `examples` | supplement: the numbers in the example figures' captions | `docci_*`, `docci/targets` |
| `clef` | supplement: the open verifier | `clef_flash` with the sources it answered |
| `fusion` | supplement: fusion without weights | `inquire_rerank_test`, `docci_*` |

`python scripts/check_export.py --all` prints one headline number per source, for a quick look at a download.

**GPT-4o.** Sec. 4 pairs our ranking with GPT-4o's published reranking on the 160 test queries with GPT-4o's released
answers. Those answers belong to INQUIRE and are not redistributed here. INQUIRE's authors shared their cache file
`openai-gpt4o20240806--nosystem-cache.npy` with us on request (their cache README lists the full set of model caches as
still to come); ask them for it, then run
`python scripts/gpt4o_answers.py --cache <file>`. It writes `external/gpt4o_answers.parquet` in this folder, outside the
data folder and ignored by git. `paper_numbers.py sec4` then checks the paired gain (+2.6, 95% CI +0.9 to +4.4); without
the file, those four numbers are skipped. GPT-4o's ranking follows INQUIRE's script (`np.flip(np.argsort(p_yes))`).
Where answers tie, numpy's argsort sets the order, which can move GPT-4o's mean by about 0.001 points across numpy
versions. The interval replays the bootstrap stream of the analysis that printed it (`qr/gpt4o.py`).

## 3. Tables and figures

`paper/build.py` writes the supplement's tables as LaTeX (`paper/out/tables/`), its plots (`paper/out/figures/`) and
every value behind them (`paper/out/values.json`), from the answers alone, in about two minutes on a CPU.
`python paper/build.py rerank docci` builds only the items named (the list is at the end of the file).
`python paper/examples.py --images <DOCCI's test images>` draws the supplement's two example figures from the answers and
DOCCI's own photos; `paper_numbers.py examples` checks the numbers their captions quote.

## Asking the questions again

Reproducing the paper does not need this. The answers come from Jev (`jev-1.14.0`, TypeSafe's hosted typed-decision
model) and from the open-weight Clef-Flash (9B), which answered the same questions. Each query's five statements were
written once by an LLM (`gpt-6.1-terra`, temperature 0) before any photo was seen; they are in `questions.parquet`, so a
new verifier can be asked exactly the same questions.

`verifier/clef_flash.py` runs Clef-Flash on a source's released questions and candidates, with the instruction and the
wording of the paper's run (the requests it builds are the run's, query for query), and writes the answers in the
release's format; `--compare` reports how far they are from the released Clef-Flash answers. It needs a CUDA GPU and
the benchmark's photos, which are not in the release: DOCCI's image folder (`docci/images.parquet` maps the photo ids to
DOCCI's file names), or the iNat24 photos with a CSV of `image_id,path` made from INQUIRE's metadata.

```bash
python verifier/clef_flash.py --source docci_s1 --images <DOCCI>/images --out docci_s1_clef.parquet --compare
python verifier/clef_flash.py --source docci_s1 --images <DOCCI>/images --limit 2 --fake-model --out test.parquet   # no GPU
```

## Data sources and licenses

The answers, questions and scores are ours. Query texts and relevance labels come from the benchmarks: INQUIRE
(CC BY-NC 4.0), DOCCI (CC BY 4.0) and Winoground (research use under its own terms; only example ids and our answers
are released, no captions, images or annotations). Because of INQUIRE's license the dataset is released under
CC BY-NC 4.0; please cite the benchmarks along with this work.
