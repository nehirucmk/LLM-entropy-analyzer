(this project is still under development!)

# llm entropy analyzer

real-time uncertainty quantification and token stream telemetry monitoring engine for large language models. built for local hardware constraints with direct logits inspection.

---

## architecture & core telemetry

standard language model interfaces output text and probabilities as black boxes. this engine hooks directly into the forward pass during inference to extract token-level uncertainty metrics on local hardware (optimized for single gpu/cpu setups).

### 1. shannon entropy ($h$)

for every generated token, the engine evaluates the full vocabulary logit vector ($v \approx 152,000$) through a softmax transformation to compute the exact shannon entropy in bits:


$$h(x) = -\sum_{i=1}^{v} p_i \log_2(p_i + \epsilon)$$

* **$h \to 0$ bits:** deterministic generation where the model is confident in the next token.
* **$h \ge 3.0$ bits:** high uncertainty state, indicating distribution flattening or potential hallucinations.

### 2. moving z-score anomaly detection

to detect sudden cognitive shifts or unexpected factual departures mid-sentence, the engine maintains a sliding window of recent entropy values to compute a rolling z-score:


$$z = \frac{h_t - \mu}{\sigma + \epsilon}$$


when $z \ge \text{threshold}$ (user-configurable, default $2.0$), the telemetry monitor triggers an automated anomaly flag.

### 3. inverse entropy control (target steering)

instead of relying on static temperature sampling (`temperature = 0.7`), the lab module allows users to dictate an exact target entropy ($h_{\text{target}}$). a built-in binary search algorithm (`solve_temperature_for_target_entropy`) reverse-solves the precise temperature ($t^*$) per token step to force the model's distribution into the desired uncertainty band:

```python
def solve_temperature_for_target_entropy(logits: torch.Tensor, target_h: float) -> float:
    t_min, t_max = 0.05, 5.0
    best_t = 1.0
    for _ in range(12):
        t_mid = (t_min + t_max) / 2.0
        probs = f.softmax(logits / t_mid, dim=-1)
        h = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
        if abs(h - target_h) <= 0.05:
            return t_mid
        if h < target_h:
            t_min = t_mid
        else:
            t_max = t_mid
        best_t = t_mid
    return best_t

```

---

## project structure

```text
.
├── app.py              # main streamlit dashboard & routing controller
├── auth.py             # sqlite security layer with bcrypt & brute-force lockout
├── generator.py        # core streaming inference engine & telemetry collector
├── lab.py              # inverse entropy target steering lab
├── metrics.py          # calculations
├── model_loader.py     # safe local weight loader (safetensors & cpu/gpu management)
├── processing.py       # sliding window statistics & z-score calculator
├── settings.py         # runtime hyperparameter configuration state
├── visuals.py          # plotly trajectory & probability distribution renderers
└── tests/
    └── leaktest.py # continuous inference memory footprint & leak verifier
    └── demo1.py 
    └── logits.py

```

---

## installation & setup

clone the repository and set up a local python virtual environment

install all the requirements: 
```
pip install -r requirements.txt

```

ensure your `.streamlit/config.toml` is configured for local execution:

```toml
[server]
address = "127.0.0.1"
headless = false

[theme]
primaryColor = "#bd9ddf"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"

```

---

## running the application

launch the web interface locally via streamlit:

```bash
streamlit run app.py

```

to verify memory stability and ensure no tensor reference leaks occur across consecutive generation passes, run the automated stress test:

```bash
python tests/leak_test.py

```

---

## security & engineering notes

* **zero external api dependency:** runs entirely offline using open-weight models (tested with qwen2.5 instruct architecture).
* **cryptographic storage:** user credentials are protected via `bcrypt` with unique salts. database queries utilize parameterized sqlite statements to completely prevent sql injection.
* **concurrency control:** database connections enforce a 10-second timeout to handle locked states gracefully during rapid testing.
* **brute-force mitigation:** consecutive failed login attempts trigger temporary account lockouts after 5 invalid tries.