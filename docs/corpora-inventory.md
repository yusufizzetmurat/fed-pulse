# External corpora inventory

_Probed: 2026-05-13T06:59:12Z_

Status legend: ✓ accessible · ✗ not accessible at this URL · ? not probed (manual fetch).

| Status | Name | Kind | Axes covered | Rows (est.) | License | URL |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ | FOMC Communication (Trillion Dollar Words) | hf_dataset | stance | — | cc-by-nc-4.0 | [link](https://huggingface.co/datasets/gtfintechlab/fomc_communication) |
| ✓ | Financial PhraseBank | hf_dataset | stance | — | ['cc-by-nc-sa-3.0'] | [link](https://huggingface.co/datasets/takala/financial_phrasebank) |
| ✗ | Gürkaynak–Sack–Swanson factor decomposition | paper_replication | factor | — | unknown | [link](https://www.federalreserve.gov/econresdata/notes/feds-notes/2015/effects-fomc-text-on-market-expectations-20151113.html) |
| ✗ | Aruoba–Drechsel narrative shocks | paper_replication | factor | — | unknown | [link](https://www.aruoba.econ.umd.edu/research/) |
| ✓ | Cieslak–Schrimpf monetary-vs-growth news | paper_replication | factor, topic | — | unknown | [link](https://sites.google.com/view/anna-cieslak/) |
| ✗ | Hansen–McMahon topic shares | paper_replication | topic | — | unknown | [link](https://stephenhansen.eu/research/) |
| ✓ | Lucca–Trebbi communication index | paper_replication | stance | — | unknown | [link](https://www.newyorkfed.org/research/staff_reports/sr357) |
| ✓ | Shapiro–Wilson FOMC tone series | fed_data_release | stance | — | unknown | [link](https://www.frbsf.org/economic-research/indicators-data/daily-news-sentiment-index/) |
| ✓ | Bauer–Bernanke–Milstein risk-appetite | paper_replication | factor | — | unknown | [link](https://www.michaeldbauer.com/research.html) |

## Per-source notes

### FOMC Communication (Trillion Dollar Words)

- **Kind:** hf_dataset
- **URL:** https://huggingface.co/datasets/gtfintechlab/fomc_communication
- **Citation:** Shah, Paturi, Chava (ACL 2023). Hand-labelled hawkish / dovish / neutral on FOMC statements, minutes, and press conferences.
- **Axes covered:** stance
- **Notes:** Primary labelled source; already ingested under the TDW alias.
- **Last-modified:** 2024-12-16T23:55:12.000Z

### Financial PhraseBank

- **Kind:** hf_dataset
- **URL:** https://huggingface.co/datasets/takala/financial_phrasebank
- **Citation:** Malo et al. (2014). 4,840 finance news sentences labelled positive / negative / neutral.
- **Axes covered:** stance
- **Notes:** Not FOMC-specific; useful as a domain-adaptive pretraining auxiliary task on the FinBERT-FedAdjacent checkpoint.
- **Last-modified:** 2025-12-15T05:51:57.000Z

### Gürkaynak–Sack–Swanson factor decomposition

- **Kind:** paper_replication
- **URL:** https://www.federalreserve.gov/econresdata/notes/feds-notes/2015/effects-fomc-text-on-market-expectations-20151113.html
- **Citation:** Gürkaynak, Sack, Swanson (IJCB 2005). 'Do Actions Speak Louder Than Words?' Target-rate vs forward-guidance shock loadings per FOMC date.
- **Axes covered:** factor
- **Notes:** Replication data historically posted on Sack's NYU page and the Federal Reserve Board's research-data archive. Manual download.
- **Last-modified:** HTTP Error 404: Not Found

### Aruoba–Drechsel narrative shocks

- **Kind:** paper_replication
- **URL:** https://www.aruoba.econ.umd.edu/research/
- **Citation:** Aruoba & Drechsel (NBER w29307). Narrative identification of monetary policy shocks from FOMC text.
- **Axes covered:** factor
- **Notes:** Posted on Aruoba's UMD page. Per-meeting shock series, csv format last time it was published.
- **Last-modified:** <urlopen error [Errno -2] Name or service not known>

### Cieslak–Schrimpf monetary-vs-growth news

- **Kind:** paper_replication
- **URL:** https://sites.google.com/view/anna-cieslak/
- **Citation:** Cieslak & Schrimpf (J. Int. Econ. 2019). Decomposition of FOMC-day price moves into monetary news and growth news.
- **Axes covered:** factor, topic
- **Notes:** Per-event labels for the FOMC release window. Posted on Cieslak's Duke page.

### Hansen–McMahon topic shares

- **Kind:** paper_replication
- **URL:** https://stephenhansen.eu/research/
- **Citation:** Hansen & McMahon (J. Int. Econ. 2016). 'Shocking Language' — LDA topic shares over FOMC statements.
- **Axes covered:** topic
- **Notes:** Per-meeting topic distributions. Hansen's Oxford / Imperial pages have hosted the replication dataset.
- **Last-modified:** <urlopen error [Errno -2] Name or service not known>

### Lucca–Trebbi communication index

- **Kind:** paper_replication
- **URL:** https://www.newyorkfed.org/research/staff_reports/sr357
- **Citation:** Lucca & Trebbi (NBER w15367 / NY Fed Staff Report 357). Continuous hawkish-dovish index built from Google-search proximity of FOMC text to anchor terms.
- **Axes covered:** stance
- **Notes:** Continuous score (not categorical). NY Fed staff-report page links a data appendix.

### Shapiro–Wilson FOMC tone series

- **Kind:** fed_data_release
- **URL:** https://www.frbsf.org/economic-research/indicators-data/daily-news-sentiment-index/
- **Citation:** Shapiro & Wilson (San Francisco Fed). FOMC-day tone series built from a constrained text-sentiment dictionary.
- **Axes covered:** stance
- **Notes:** SF Fed publishes the tone series openly. Verify which subseries covers FOMC text vs general news.

### Bauer–Bernanke–Milstein risk-appetite

- **Kind:** paper_replication
- **URL:** https://www.michaeldbauer.com/research.html
- **Citation:** Bauer, Bernanke & Milstein (NBER 2023). Risk-appetite channel of monetary policy; FOMC-day shock decomposition.
- **Axes covered:** factor
- **Notes:** Posted on Bauer's UC Irvine page.

