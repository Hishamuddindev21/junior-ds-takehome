# Delivery Time Prediction - Take-Home Assignment

## What I Built

A machine learning model that predicts how long deliveries will take based on distance, traffic, weather, and other factors. The model achieves an R² of 0.89, meaning it explains about 89% of the variation in delivery times.

---

## Quick Results

**Model Performance:**
- Average prediction error: **3.7 minutes**
- R² Score: **0.89** (pretty good!)
- No overfitting detected

**Key Finding:** Weather and traffic are the biggest factors affecting delivery time. Stormy weather adds ~18 minutes, while low traffic saves ~15 minutes.


## How to Run

```bash
# Install stuff
pip install -r requirements.txt

# Generate data
python src/generate_data.py

# Run the analysis
jupyter notebook notebooks/solution.ipynb
```

---

## Part 1: Data Cleaning

**Problems I found:**
1. Missing driver ratings (5% of data) → filled with median value (4.0)
2. Negative package weights (2% of data) → took absolute values

**New features I created:**
- Hour of day (0-23)
- Day of week (Monday=0, Sunday=6)
- Month (1-12)

These help capture patterns like "rush hour deliveries take longer" or "weekends are faster."


## Part 2: Pipeline

Built a proper sklearn pipeline to avoid data leakage:

```python
Pipeline([
    ('preprocessing', ColumnTransformer([
        ('scale_numbers', StandardScaler(), numeric_features),
        ('encode_categories', OneHotEncoder(), categorical_features)
    ])),
    ('model', LinearRegression())
])
```

**Important:** The preprocessing only learns from training data, then applies the same transformations to test data. This prevents the model from "cheating."

**Features used:**
- Numeric: distance, weight, driver rating, hour, day, month
- Categorical: weather (sunny/rainy/etc), traffic (low/medium/high)


## Part 3: Results

### Metrics Breakdown

| Metric | Value | What it means |
|--------|-------|---------------|
| MAE | 3.71 min | Average error is about 4 minutes |
| RMSE | 4.75 min | Typical deviation from actual time |
| R² | 0.89 | Model explains 89% of the variance |

### What drives delivery time?

1. **Stormy weather**: +18 minutes (ouch!)
2. **Low traffic**: -15 minutes (nice!)
3. **Distance**: +11 minutes per km
4. **Rain**: +9 minutes
5. **Medium traffic**: -10 minutes

The model makes sense - bad weather slows things down, light traffic speeds things up, and longer distances take more time.


## Part 4: Questions

### Question 1: You might have noticed rows with negative package weights. If you found that 25% of the dataset had negative weights, would you drop them? If not, what would you do instead?

**Answer: No, I wouldn't drop them.**

Dropping 25% of your data is a bad idea - you lose too much information and your model gets weaker.

**What I'd do:**

```python
# Create a flag for negative weights
df['had_negative_weight'] = (df['weight'] < 0).astype(int)

# Fix the actual weights
df['weight'] = df['weight'].abs()
```

Then use both columns as features.

**Why this works:**
- Keeps all the data
- Fixes the obviously wrong values (negative weights don't make sense)
- The flag captures any pattern - maybe certain delivery types have more data errors, or maybe it means something specific we don't know about yet
- If the pattern is meaningful, the model will use it. If it's random noise, it won't hurt anything.

In my solution, only 2% of weights were negative so I just took absolute values. But with 25%, you'd want that extra flag to catch any patterns.

---

### Question 2: Imagine the traffic_level data comes from a paid API that costs us money every time we call it. How would you determine if this feature is 'worth' the cost?

**Answer:**

**Step 1: See if it actually helps**

Build two models and compare:
```python
# With traffic
model_with_traffic.fit(X_train, y_train)
mae_with = mean_absolute_error(y_test, predictions_with_traffic)

# Without traffic
model_no_traffic.fit(X_train_no_traffic, y_train)
mae_without = mean_absolute_error(y_test, predictions_no_traffic)

# How much better is it?
improvement = mae_without - mae_with
```

**Step 2: Calculate what it costs**

```
$0.01 per API call
1,000 deliveries per day
= $10/day = $3,650/year
```

**Step 3: Figure out what you get for that money**

If the model without traffic is 1.5 minutes less accurate on average:
- More wrong ETAs → more angry customers calling support
- Say 10 extra calls per day at $5 each to handle = $50/day saved
- That's $18,250/year saved

So you pay $3,650 to save $18,250 → worth it.

**Step 4: Decide**

Keep the feature if the benefits are bigger than the cost. In my model, traffic has a huge effect (low traffic saves 15 minutes), so it's definitely worth keeping.


