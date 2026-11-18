# LatentWAV documentation

Start with the project [README](../README.md) for a simple explanation of the
whole system. Use the pages below to understand one part at a time.

Experiments, including failed and ongoing runs, are recorded in the
[experiment log](experiments/README.md).

## Component index

| Component | Simple description | Documentation |
|---|---|---|
| NanoWM | Dreams possible future pictures | [NanoWM](components/nanowm.md) |
| IDM | Guesses actions for reaching a future picture | [IDM](components/idm.md) |
| LeWM | Predicts what an action will do | [LeWM](components/lewm.md) |
| Verifier | Finds the action LeWM understands least | [Verifier](components/verifier.md) |
| Data pipeline | Prepares PushT examples for training | [Data pipeline](components/data-pipeline.md) |
| Online policy | Connects the models and chooses actions | [Online policy](components/online-policy.md) |
| Collection and evaluation | Saves experience and checks improvements | [Collection and evaluation](components/collection-evaluation.md) |

For the end-to-end design and implementation status, see
[COMPONENTS.md](COMPONENTS.md). For a non-technical introduction, see the
[project README](../README.md).
