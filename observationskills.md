You are working inside the Jupyter notebook `Electrical_Load_Forecasting_MLP.ipynb`. It is already
fully executed with real outputs (plots, tables, printed values).

Before writing anything, read the style/voice rules in this file and follow them exactly:
`/Users/shayanazmi/college/college/sem5/ann_&_deeplearning/notbookwriting-skill.md`

Your task: go through the notebook cell by cell, in order, and after EVERY cell that produces a
visual (chart/plot/heatmap/figure) or a numerical result (printed metric, table, dataframe), insert
a new markdown cell directly below it titled "**Observation:**" containing 2-4 sentences that:

1. State what the output actually shows — reference the real numbers/shapes/trends visible in that
   specific output (e.g. actual R2 values, actual skew value, actual outlier %, which curve is on
   top, which cell in a heatmap is darkest), not a generic description of what the plot type usually
   shows.
2. Explain WHY it looks that way, grounded in the dataset or the model choice made earlier in the
   notebook (e.g. "this happens because sigmoid saturates near |x|>2 on our standardized inputs" or
   "this spike is the summer-heatwave demand surge identified in Section 1.3.9" or "the gap between
   train and val loss here reflects the modest 64-32-16 architecture relative to ~32k training rows").
   Do not just restate the observation without a causal reason.
3. Match the notebook's existing voice AND the style rules from notbookwriting-skill.md: technical,
   concise, non-repetitive, no fluff, no restating the section's already-written markdown -- add NEW
   insight the reader doesn't already have.
4. Where relevant, tie the observation back to how it affects a later decision in the notebook (e.g.
   activation choice in Section 3, loss choice in Section 4, final model choice in Section 8) rather
   than treating each output as isolated.

Rules:
- Read notbookwriting-skill.md FIRST and apply its formatting/tone/terminology conventions to every
  Observation cell you write. If it conflicts with anything below, the skill file wins.
- Do not modify or delete any existing cell (code or markdown) or re-run the notebook -- only ADD new
  markdown "Observation" cells using the values already present in the existing outputs.
- Read the actual printed/rendered output of each cell (not just the code) before writing its
  observation -- the numbers you cite must match what's actually shown.
- Skip purely structural cells (imports, print-only scope/status statements, hyperparameter-table
  echoes, shape summaries with no interpretive value) -- only add observations where there's a real
  visual or a result worth interpreting.
- Keep each Observation cell short (2-4 sentences) -- do not turn this into a full report.
- Do not change section numbering or headers; your Observation cells sit as sub-content immediately
  under the output they refer to.

Work through the notebook sequentially, section by section (1 through 8), and save the notebook
in place when done.