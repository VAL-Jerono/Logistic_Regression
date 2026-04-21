# Customer Churn Prediction — Logistic Regression

A telecom company is bleeding customers. Quietly, one by one, people are cancelling
their subscriptions — and the company only finds out after they are gone.

The question this notebook answers: **can we see it coming?**

---

## The Business Problem

Every month, some customers stay and some leave. The ones who leave are called
**churners**. If you could identify them *before* they cancelled, you could
intervene — a call, a discount, a better plan. You could protect revenue instead
of mourning it.

This is a supervised binary classification problem. Each customer either churned
(`Yes`) or did not (`No`). The model's job is to learn the difference from
historical data, then flag at-risk customers before they walk out.

---

## Why Not Just Use Linear Regression?

The natural instinct is to reach for the familiar tool. Predict a number — you
know how to do that. But churn is not a number. It is a verdict. A customer either
left or they did not.

Force linear regression onto this problem and it will produce answers like `1.4`
or `-0.2`. Those are not probabilities. They are nonsense.

**Logistic regression** wraps the same linear equation inside a sigmoid function:

$$P(\text{churn} = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k)}}$$

The sigmoid squeezes any real number — positive, negative, enormous — into the
interval (0, 1). Every output is a valid probability. A customer with score `0.82`
carries high churn risk. A customer at `0.11` is likely to stay.

Internally, the model still works with a linear equation — but in **log-odds
space**:

$$\log\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k$$

The sigmoid is just the transformation that brings you back to probabilities when
you need a decision.

---

## Before the Model: Understanding the Data

A model trained on data it does not understand is a model you cannot trust.

The notebook begins with a full exploratory analysis: structure, summary statistics,
a missing value audit, and a class balance check. That last one matters more than
beginners expect. If 93% of customers in the dataset did not churn, a model that
always says "No" achieves 93% accuracy while being completely useless. Accuracy
flatters bad models on imbalanced data. This notebook uses better metrics.

---

## The Step Everyone Skips: Casting the Outcome to a Factor

`churn` arrives in the dataset as numeric — `0` and `1`. That looks fine. It is not.

When R's `glm()` sees a numeric outcome, it fits **linear regression**. The sigmoid
is never applied. The log-likelihood is never maximised. The entire machinery of
logistic regression is bypassed in silence.

```r
df$churn <- factor(df$churn, levels = c(0, 1), labels = c("No", "Yes"))
```

One line. It tells `glm()`: *this is a category, not a measurement. Apply the
binomial family. Use the logit link.* Every logistic regression pipeline begins here.

---

## Choosing Which Features to Keep

Not every column deserves to be in the model.

Irrelevant features add noise. Correlated features fight each other — when two
predictors tell the same story, the model cannot decide how much credit to give each
one, and both coefficients become unstable and uninterpretable.

The notebook uses **point-biserial correlation** to score each feature against the
churn outcome. Features below the `|r| > 0.10` threshold carry almost no linear
signal and are dropped. A correlation heatmap then checks whether any surviving
features are too similar to each other.

The dataset contains `loglong` and `lninc` — log-transformed versions of `longmon`
and `income`. Including both the original and the log transform is like counting
the same evidence twice. One of each pair is dropped.

What remains is a lean set of predictors: everything that matters, nothing that
clutters.

---

## Normalisation: The Step That Determines Whether Training Works

`income` ranges from near zero to over a thousand. A binary contract flag sits at
zero or one. The model sees both in the same equation.

R's `glm()` fits logistic regression using **Iteratively Reweighted Least Squares
(IRLS)** — an iterative algorithm that nudges coefficients toward the maximum of
the log-likelihood surface one step at a time. When features live on wildly
different scales, those steps are uneven: the optimiser either crawls on
large-scale features or overshoots on small ones. Convergence is slow, unreliable,
or both.

Z-score standardisation solves this:

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

After scaling, every feature has mean zero and standard deviation one. The optimiser
works in a uniform space. The shape of every distribution is preserved — only the
units change.

**The rule that cannot bend:** the scaler is fitted on training data only, then
applied identically to both sets. Fitting on the full dataset lets test-set
information leak into training. Metrics look better than they are. The model is
evaluated on data it has already seen, in disguise. This is called **data leakage**,
and it is how models that perform beautifully in the notebook fail quietly in
production.

---

## Training the Model

Once the data is correctly prepared, the modelling step is brief:

```r
model <- glm(
  churn ~ .,
  data   = train_scaled,
  family = binomial(link = "logit")
)
```

The hard work was everything before this line.

`glm()` returns coefficients in **log-odds units**. A positive coefficient means
that feature pushes the probability of churn upward. A negative coefficient
suppresses it. The magnitude tells you how much.

Exponentiate the coefficients and they become **odds ratios** — which are far easier
to communicate:

> *A one-standard-deviation increase in long-distance call minutes is associated
> with 50% higher odds of churning, holding all other features constant.*

A **forest plot** visualises every coefficient alongside its 95% confidence
interval. If an interval bar crosses zero, the effect may not be statistically
reliable — the model is uncertain whether that feature moves churn risk at all.

---

## Evaluating the Model

### Log-Loss: What the Model Actually Minimised

During training, logistic regression did not try to maximise accuracy. It minimised
**log-loss** — binary cross-entropy:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \Bigl[ y_i \cdot \log(\hat{p}_i) + (1 - y_i) \cdot \log(1 - \hat{p}_i) \Bigr]$$

Log-loss punishes confident wrong predictions harshly. A model that says
P(churn) = 0.99 for a customer who then stays has made a confident mistake. The
penalty is large. This pressure is what teaches the model to be **well-calibrated**:
not just directionally correct, but appropriately uncertain when the signal is weak.

The random baseline is `0.6931` — the log-loss of a coin flip. A useful model
scores below it. A model scoring above it is worse than knowing nothing.

### AUC-ROC: Can the Model Rank Risk?

The ROC curve asks a different question: if you pick one random churner and one
random non-churner, what is the probability the model assigns the higher risk score
to the churner?

That probability is the **AUC** — Area Under the Curve. A random model scores
`0.5`. A perfect model scores `1.0`. AUC does not care about the classification
threshold at all — it measures pure ranking quality, which is why it holds up on
imbalanced datasets where accuracy misleads.

### The Confusion Matrix: A Map of the Errors

At threshold `0.5`, every predicted probability becomes a label. The confusion
matrix tallies the four outcomes:

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | True Negative ✓ | False Positive — unnecessary intervention |
| **Actual Yes** | False Negative — missed churner | True Positive ✓ |

In a churn problem, **False Negatives are expensive**. A missed churner represents
permanently lost revenue. A false positive costs one retention call. The business
context determines which error to minimise — and the threshold on the ROC curve
can be shifted to reflect that priority.

---

## What the Numbers Tell You

The notebook closes with a complete metrics table — accuracy, precision, recall,
specificity, F1-score, log-loss, and AUC-ROC — with plain-language interpretations
for each, and a final printed summary consolidating every decision made across the
full pipeline.

---

## Running the Notebook

Place `ChurnData.csv` in your working directory, then:

```r
rmarkdown::render("Churn_LogisticRegression_Teaching.Rmd")
```

**Required packages:**

```r
install.packages(c(
  "tidyverse", "corrplot", "scales",
  "caret", "pROC", "MLmetrics",
  "broom", "knitr", "kableExtra"
))
```

Output is a self-contained HTML file with a floating table of contents, numbered
sections, and all code visible — every line annotated, every decision explained.

---

## Where to Go Next

The notebook ends with working stubs for five natural extensions:

- **Class weights** — penalise missed churners more heavily than false alarms
- **Threshold optimisation** — find the cutoff that fits your business cost structure
- **LASSO / Ridge regularisation** — built-in feature selection via `glmnet`
- **Stepwise selection by AIC** — let `step()` prune the model greedily
- **Non-linear comparison** — `randomForest` and `xgboost`, to test whether the linear assumptions are enough

---

## Who This Is For

- Students encountering logistic regression for the first time who want to understand the *why*, not just the *how*
- Practitioners who have used `glm()` but never felt confident interpreting the output
- Instructors who want a ready-made, heavily commented classroom notebook

By the end, the pipeline you have built — correlation-filtered features, leak-free normalisation, properly cast outcome, rigorous multi-metric evaluation — transfers directly to any binary classification problem you will ever face.

---






*Built with R · tidyverse · caret · pROC · MLmetrics · broom*

---

*Everything here transfers directly to any binary classification problem you will
ever face. The dataset changes. The pipeline does not.*