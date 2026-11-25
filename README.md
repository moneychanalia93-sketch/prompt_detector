# Prompt Detector

Classifies a prompt as human-written or synthetic (LLM-generated) using
stylistic heuristics and, optionally, GPT-2 statistics (perplexity and
GLTR-style token-rank fractions), fed to a scikit-learn `GradientBoostingClassifier`.

## Setup (Windows, PowerShell)

```powershell
cd prompt_detector
py -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```

Heuristics-only mode does not need torch/transformers:

```powershell
pip install scikit-learn numpy joblib pytest streamlit
```

## Train

```powershell
python train.py            # heuristics + GPT-2 features (downloads gpt2 on first run, ~500 MB)
python train.py --no-lm    # heuristics only, fast baseline
```

Prints a classification report on a 20% stratified holdout and writes `detector.joblib`
(model + feature names). Detection automatically uses GPT-2 only if the saved model was trained with it.

Data format: `data/human.txt` and `data/synthetic.txt`, one prompt per line.
A literal `\n` in a line becomes a newline (for bulleted prompts).

## Detect

```powershell
python detect.py "You are a senior engineer. Provide a comprehensive, step-by-step guide to ..."
python detect.py "some prompt" --json
```

Output: `probability_synthetic`, `verdict` (synthetic > 0.7, human < 0.3, otherwise
`uncertain`; `uncertain (too short)` under 15 words), and the top signals.
The signals are readable rules over the features, not an exact attribution of the model's decision.

## Streamlit demo

```powershell
streamlit run app.py
```

## Tests

```powershell
python -m pytest
```

## Caveats

- The starter dataset is small (~200 hand-written lines) and the synthetic class is stylistically
  narrow. Scores on the holdout are optimistic and do not reflect real-world accuracy.
- Short prompts (< 15 words) carry little signal; the verdict is capped at "uncertain (too short)".
- Adversarial editing defeats this easily: lowercasing, adding typos, removing stock phrases, or asking an LLM to "write casually".
- Non-native writers and people who write formal, template-like prompts (or paste prompts from guides) can be flagged as synthetic.
- GPT-2 perplexity is also low for famous or repeated text.
- Do not use this to make decisions about people.

## Better data

Train on real prompts instead of the starter set:

- Human: [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) and
  [WildChat](https://huggingface.co/datasets/allenai/WildChat) user turns (gated or licensed; read the terms).
  Note that these include some prompts that users themselves pasted from LLM output.
- Synthetic: generate prompts with several LLMs (vary models, temperature, and instructions, including
  "write like a casual user") so the detector does not learn one model's style.
- Keep a holdout split by source/model, not just random rows, to measure generalization.
