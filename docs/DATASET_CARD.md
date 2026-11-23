---
license: cc-by-nc-4.0
language:
- en
pretty_name: Questioning Image Retrieval, verifier answers
size_categories:
- 10M<n<100M
task_categories:
- visual-question-answering
tags:
- image-retrieval
- reranking
- vqa
- inquire
- docci
- winoground
configs:
{{configs}}
---

# Questioning Image Retrieval: the verifier's answers

Every answer behind the CVPR 2027 submission *Questioning Image Retrieval: Cheap Verifiers Do More Than Rerank*. For each query an LLM wrote five statements that a matching photo must satisfy; a fast typed-decision
verifier then answered seven questions about every photo the search returned: the whole query as a yes or no question,
the five statements, and a 1-to-5 score of how well the photo matches. Every table and figure of the paper is a
function of these answers, so the paper can be checked, re-scored and extended without calling the verifier, and a new
verifier can be asked exactly the same questions.

{{stats}}

The code that recomputes every number of the paper from these tables is in the paper's repository (`scripts/fetch_data.py`
downloads this dataset and checks each file's sha256; `paper/paper_numbers.py` recomputes every number these tables
determine and sets each against the paper).

## Loading

```python
import pandas as pd
base = "hf://datasets/{{repo_id}}"
answers = pd.read_parquet(f"{base}/docci_s2/answers.parquet")      # one row per (photo, question)
# or: datasets.load_dataset("{{repo_id}}", "answers", split="docci_s2")
```

## Sources

| source | benchmark, queries | photos per query |
|---|---|---|
| `docci_s1` .. `docci_s4`, `docci_all` | DOCCI test, 500 descriptions: the first 1 to 4 sentences to arrive, and all of them | SigLIP2's top 100 |
| `docci_heldout` | DOCCI test, 500 other descriptions, the whole text as one query (the search reads its first 64 tokens) | SigLIP2's top 200 |
| `docci_deep200_s1` .. `s4` | the same 500 descriptions, looking further: the search's top 200, then the photos the answers chose | about 235 |
| `inquire_rerank_test`, `inquire_rerank_val` | INQUIRE-Rerank, 160 and 40 queries | INQUIRE's fixed 100 |
| `inquire_full_vith_lc_test`, `_val` | INQUIRE-Fullrank (5 million photos), 200 and 50 queries | CLIP ViT-H/14's top 500 |
| `inquire_full_siglip2_lc_test`, `_val` | INQUIRE-Fullrank, 200 and 50 queries | SigLIP2's top 500 |
| `clef_flash` | the open Clef-Flash (9B) answering the same questions; `source` names the run | 100 per query |
| `winoground` | Winoground, 400 examples; one yes or no question per caption and image; `mode`: the run, the text-only control and shuffled captions | 4 per example |
| `bank_inquire_rerank_test`, `bank_inquire_full_vith_lc_val`, `bank_docci_s1` | the larger question set of Sec. 4 (15.7 questions per photo in seven families), asked plainly and with an LLM-written brief, with the earlier run's seven answers and search scores on the same photos | 100, 500, 100 per query |
| `bank_inquire_wide_test`, `bank_inquire_wide_val` | INQUIRE-Fullrank, 200 and 50 queries: the larger question set asked plainly and with the brief of every photo checked for the query in four earlier runs (Table 4, "report uncertainty") | about 900: CLIP ViT-H's top 500, SigLIP2's top 500, and the photos of two runs that searched with 45 LLM-written texts per query |

## Tables

| table | one row per | columns |
|---|---|---|
| `queries` | query | `source`, `benchmark`, `subset`, `split`, `query_id`, `query` (the text the verifier read), `relevant` (image ids), `meta` (JSON: INQUIRE's supercategory, category and iconic group) |
| `questions` | question | `query_id`, `question_id` (`holistic`, `sub0`..`sub4`, `relevance`), `role`, `kind` (`noul`: yes or no; `score`: 1 to 5), `text`, `weight` (statements: 0.5 nice to have to 1.5 essential), `levels` (the 1-to-5 labels) |
| `candidates` | photo checked | `query_id`, `image_id`, `search_rank`, `search_score`, `siglip2_score` |
| `answers` | photo and question | `query_id`, `image_id`, `question_id`, `p_yes`, `p_levels`, `expected` (1-to-5 answer, expected value scaled to [0, 1]), `verifier`, `input_tokens`, `error` |
| `docci_targets` | DOCCI target | `query_id`, `row`, `image_id`, `docci_example`, `sentences`, `arrival` (the seeded order in which sentences arrive), `n_sentences` |
| `docci_images` | DOCCI test photo | `image_id`, `docci_example`, `file_name` |
| `firststage` | (query, rank) | the pool encoder's top 1,000 for INQUIRE-Fullrank: `query_id`, `rank`, `image_id`, `score`, `siglip2_score` |
| `steered` | (query, photo) | the 100 photos the answers chose, in the steered search's order |
| `taxa` | INQUIRE photo | iNat24's category of every photo in the Fullrank first stages: `image_id`, `category` (the folder name: category id, kingdom, phylum, class, order, family, genus, species). Sec. 7 uses it to tell a look-alike's species from the target's; the method never sees it |
| `streams` | (query, search) | every saved SigLIP2 search of an INQUIRE-Fullrank query (Table 4's feedback rows): `query_id`, `position` (the order the analysis read the queries in), `stream` (`q`; an LLM text alone, e.g. `names0`; anchored to the query, `q+names0`; a family's texts with the query, `q+names`; every text, `q+all`; fitted on the verifier's first answers, `q+feedback`), `text`, `ids` (top 200) |
| `steer_searches` | (query, search) | Table 4's steered search on INQUIRE-Fullrank: `split`, `query_id`, `position` (the analysis's order of the queries), `checked` (photos of CLIP ViT-H's ranking checked before the search: 100 or 500; 0 for the plain search), `search` (`text`: SigLIP2's search for the query; `answers`: the query's vector moved along a direction fitted on the checked photos' answers; `labels`: fitted on their labels, a control), `ids` (top 1,000 of the 5 million photos) |
| `ledgers` | (run, role) | requests, tokens and prices of the INQUIRE-Rerank test run, for the verifier and for the LLM that wrote the statements |
| `gpt4o_pool` | (query, rank) | the photos GPT-4o reranks in the INQUIRE paper, CLIP ViT-H's top 100 for the query as written: `query_id`, `inquire_query_id`, `split`, `rank`, `image_id`, `image_path` (iNat24's relative path). GPT-4o's answers are INQUIRE's and are not included |

The larger question set has four tables of its own (`bank_queries`, `bank_photos`, `bank_questions`, `bank_answers`): each
question's family (W whole query in three wordings, R requirements, C details, N the likely alternative, M match or
near-miss, V 1-to-5 match, X what is wrong, F descriptors), its options, and each answer as a value on a 0-to-1 scale
where 1 speaks for the question's target, with every option's probability. `bank_queries.brief` is what the verifier
was told when asked with the brief (the instruction, the query and the LLM's background); asked plainly, it was told
only "You are judging one candidate image returned by an image search engine." The wide sources' photos are in
`bank_wide_photos`, which adds each photo's rank in the four earlier runs' pools and the search score each run stored,
SigLIP2's similarity to the query in all four (`rank_vith`, `score_vith`, `rank_siglip2`, `score_siglip2`, `rank_xpool`,
`score_xpool`, `rank_spool`, `score_spool`; empty when the photo was not in that pool: `vith` and `siglip2` are the
query's own top 500 with each encoder, `xpool` and `spool` the two runs with many searches); their answers keep one row
group per query.

The score of the paper, for one photo: $S = \exp(\sum_j w_j \log \max(s_j, 0.02) / \sum_j w_j)$ over the statements and
$f = 0.8\,(0.4\,h + 0.4\,S + 0.2\,r) + 0.2\,c$, with $h$ the whole query's `p_yes`, $s_j$ and $w_j$ the statements'
`p_yes` and `weight`, $r$ the score question's `expected`, and $c$ the `search_score` min-max scaled over the pool. A
photo with `error` keeps only $0.2\,c$. Ties go to the better `search_rank`.

Search scores: SigLIP2 so400m/16-384 with the query lowercased (INQUIRE-Rerank test, DOCCI; `siglip2_score` keeps the
score for the query as written on INQUIRE-Rerank test). INQUIRE-Rerank validation kept the query as written.
INQUIRE-Fullrank: the encoder that built the pool, with its own rank and score. DOCCI with sentences: one SigLIP2 vector
per sentence, averaged.

## Images

No image is included. Image ids are the benchmarks' own: INQUIRE's image id (iNaturalist 2024 images, available from
INQUIRE's release), DOCCI `i<row>` of the test split (`docci_images` gives DOCCI's example id and file name), and
Winoground's example id with caption and image 0 or 1.

## Licenses and attribution

The answers, the statements and the scores are released under CC BY-NC 4.0, the license of INQUIRE, whose query texts
and relevance labels are included. DOCCI's descriptions (CC BY 4.0) are included as queries and in `docci_targets`.
Winoground is shared for research use under its own terms; only its example ids and our answers are included, with no
caption, image or annotation. Please cite the benchmarks with this work:

- INQUIRE: Vendrow et al., *INQUIRE: A Natural World Text-to-Image Retrieval Benchmark*, NeurIPS 2024 Datasets and Benchmarks.
- DOCCI: Onoe et al., *DOCCI: Descriptions of Connected and Contrasting Images*, ECCV 2024.
- Winoground: Thrush et al., *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality*, CVPR 2022.

## Citation

Anonymous. *Questioning Image Retrieval: Cheap Verifiers Do More Than Rerank*. CVPR 2027 submission.
