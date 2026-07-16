# APSS-SLU Analysis — Run Guide

This folder contains the three Python scripts that produce the project deliverables:
the **Validation Report**, the **Data Analysis Report**, and the **three figures** used in the paper.

---

## 1. Setup (once)

You need Python 3.8+ and these libraries:

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

The scripts read two CSV files that must sit in the **same folder** as the scripts:

| Role | File name |
|------|-----------|
| Round 1 (test, 93 responses, **with emails**) | `Adapted Perceived Stress Scale for Sri Lankan Undergraduates (APSS-SLU) (Responses) - Form responses.csv` |
| Round 2 (retest, 76 responses, with emails) | `2nd Round Responses.csv` |

> **Important:** Round 1 **must** keep its `Email address` column populated. Test–retest reliability
> works by matching the same person across the two rounds by email. If you ever swap in an
> anonymized Round-1 file (emails blank), the scripts will *not* crash — they now print a clear
> "SKIPPED" message and drop the test–retest section / Figure 1 instead.

---

## 2. Order of execution

The three scripts are **independent** — none depends on another's output, so any order works.
The logical order (matching the paper) is:

```bash
python 01_validation_report.py      # -> validation_report.txt
python 02_data_analysis_report.py   # -> data_analysis_report.txt
python 03_generate_figures.py       # -> figures/  (6 files: 3 PNG + 3 PDF)
```

Each script prints its report to the console **and** saves it to a file.

---

## 3. What each script does

### `01_validation_report.py` → `validation_report.txt`

Answers the assignment's reliability/validity requirement. Internally it:

1. **Scores** every response using PSS-14 rules — 7 negatively-worded items are scored directly,
   the 7 positively-worded items are reverse-scored (`4 - x`), then summed to a **Total** (0–56).
2. **Internal consistency** — computes **Cronbach's α** for both rounds (R1 = 0.546, R2 = 0.427),
   plus per-item statistics: corrected item-total correlation and "α-if-item-deleted". This shows
   *why* α is modest — a few reverse-scored items discriminate poorly.
3. **Construct validity** — runs **PCA** (Principal Component Analysis) on the standardized 14 items,
   reports eigenvalues, variance explained, and a 2-factor loading table mapping onto the theorized
   *Helplessness* and *Self-Efficacy* subscales.
4. **Test–retest reliability** — matches the 76 people who answered both rounds **by email**, then
   reports **Pearson r = 0.935**, Spearman ρ, and a paired-t test (p = 0.212 → no systematic drift):
   excellent temporal stability.

### `02_data_analysis_report.py` → `data_analysis_report.txt`

The demographics-and-hypothesis-testing analysis on the Round-1 sample (N = 93):

1. **Demographics** — counts/percentages for gender, year, faculty, employment, living, sleep, age.
2. **Descriptive statistics** — mean, SD, median, skew, kurtosis of the Total score, a Shapiro–Wilk
   normality check, and the same descriptives broken down by every subgroup.
3. **Inferential statistics** — one test per demographic factor, each with an effect size:
   - **Gender** — Welch t-test → t(90) = −2.17, **p = 0.034** (females higher).
   - **Year of study** — one-way ANOVA → F(3,89) = 6.01, **p = 0.001** (stress rises to senior years),
     with Bonferroni-corrected post-hoc pairwise tests.
   - **Part-time work** — Welch t-test → not significant.
   - **Living arrangement** — one-way ANOVA → not significant.
   - **Sleep duration** — one-way ANOVA → F(2,89) = 3.23, **p = 0.044** (the single "More than 8 hours"
     respondent is excluded as a singleton cell).
4. A summary table and an overall interpretation section.

### `03_generate_figures.py` → `figures/` (PNG @ 300 dpi + PDF for LaTeX)

Generates the three paper figures. It re-scores the data the same way, then draws:

- **`fig_retest`** — *how we got Figure 1.* Merges the two rounds by email (76 matched people),
  plots each person's **Round-1 Total (x) vs Round-2 Total (y)**, overlays the regression line and a
  dashed line-of-equality, and annotates **Pearson r = 0.935**. Tight clustering on the equality line
  = stable instrument.
- **`fig_year`** — *how we got Figure 2.* For each year of study it computes the **mean Total and the
  standard error of the mean (SEM)**, draws a bar per year with ±1 SEM error bars, colours juniors
  (Y1–Y2) grey and seniors (Y3–Y4) navy, and marks the p = 0.001 ANOVA result — visualising the
  junior-to-senior escalation.
- **`fig_itemtotal`** — *how we got Figure 3.* For each of the 14 items it computes the **corrected
  item-total correlation** (the item vs. the sum of the *other* 13 items) and draws a horizontal bar,
  colouring direct items navy and reverse-scored items red, with a dashed adequacy threshold at
  r = 0.30. The short red bars explain the modest Cronbach's α.

---

## 4. Expected results (sanity check)

If your run matches these, everything is wired correctly:

| Output | Value |
|--------|-------|
| Cronbach's α (Round 1) | 0.546 |
| Test–retest matched N / Pearson r | 76 / 0.935 |
| Paired-t drift | t = 1.26, p = 0.212 (n.s.) |
| Gender | t(90) = −2.17, p = 0.034 |
| Year of study | F(3,89) = 6.01, p = 0.001 |
| Sleep duration | F(2,89) = 3.23, p = 0.044 |
| Figures produced | 6 files in `figures/` (3 PNG + 3 PDF) |
