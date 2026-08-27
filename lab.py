import streamlit as st
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from typing import Generator, Dict, Any


def solve_temperature_for_target_entropy(logits: torch.Tensor, target_h:float)->float:
    #binary search 
    t_min, t_max = 0.05, 5.0
    best_t = 1.0
    for _ in range(12):
        t_mid = (t_min+t_max)/2.0
        probs = F.softmax(logits/t_mid, dim=-1)
        h= -torch.sum(probs*torch.log2(probs+1e-12)).item()
        if abs(h- target_h)<=0.05:
            return t_mid
        if h<target_h:
            t_min = t_mid
        else:
            t_max = t_mid
        best_t= t_mid
    return best_t

def generate_with_custom_entropy(
        prompt: str,
        target_entropy: float,
        max_tokens: int,
        model: Any,
        tokenizer: Any
)-> Generator[Dict[str, Any], None, None]:
    device= next(model.parameters()).device
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    past_key_values = None
    stop_tokens = {tokenizer.eos_token_id}

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs=model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values
            logits= outputs.logits[:, -1, :].squeeze(0)

        dyn_temp = solve_temperature_for_target_entropy(logits,target_entropy)
        probs = F.softmax(logits/dyn_temp, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        token_id_int =next_token_id.item()
        if token_id_int in stop_tokens:
            break

        token_str = tokenizer.decode(token_id_int, skip_special_tokens=True)
        actual_h=-torch.sum(probs*torch.log2(probs+1e-12)).item()

        yield{
            "token": token_str,
            "actual_h": actual_h,
            "target_h": target_entropy
        }

        input_ids= torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

def render_entropy_lab(model_getter) -> None:
    st.markdown("##### Entropy Laboratory")

    col_left, col_right = st.columns([1.1, 1], gap="medium")

    with col_left:
        prompt_text = st.text_area(
            "Input Prompt",
            value="Explain quantum superposition in two concise sentences.",
            height=70
        )

        ctrl_col1, ctrl_col2 = st.columns([1.4, 1])
        with ctrl_col1:
            target_h = st.slider("Target Entropy (bits)", 0.3, 3.5, 1.5, 0.1)
        with ctrl_col2:
            max_tok = st.slider("Max Tokens", 20, 150, 60, 10)

        start_btn = st.button("Generate with Target Entropy", type="primary", use_container_width=True)

        st.markdown("###### Model Response Stream")
        response_box = st.container(height=210)
        text_placeholder = response_box.empty()
        text_placeholder.caption("Awaiting generation prompt...")

    with col_right:
        st.markdown("###### Real-Time Telemetry Tracking")
        top_row = st.columns([1, 1, 1.2])
        m_target = top_row[0].empty()
        m_actual = top_row[1].empty()
        m_status = top_row[2].empty()

        chart_placeholder = st.empty()

        m_target.metric("Target (H)", f"{target_h:.1f}")
        m_actual.metric("Actual (H)", "--")
        m_status.info("Idle")
        chart_placeholder.caption("Trajectory plot will render during stream execution.")

    # 2. Execution loop bounded to the pre-allocated placeholders
    if start_btn:
        if not prompt_text.strip():
            st.warning("Please enter a valid prompt.")
            return

        model, tokenizer = model_getter()
        stream = generate_with_custom_entropy(
            prompt=prompt_text,
            target_entropy=target_h,
            max_tokens=max_tok,
            model=model,
            tokenizer=tokenizer
        )

        full_text = ""
        log_h = []

        for packet in stream:
            full_text += packet["token"]
            log_h.append(packet["actual_h"])
            text_placeholder.markdown(full_text + "▌")

            m_actual.metric("Actual (H)", f"{packet['actual_h']:.2f}")

            diff = abs(packet["actual_h"] - target_h)
            if diff > 0.6:
                m_status.error("High Variance")
            elif diff > 0.3:
                m_status.warning("Tracking")
            else:
                m_status.success("Locked")

            steps = list(range(1, len(log_h) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=steps,
                y=log_h,
                mode="lines+markers",
                name="Observed H",
                line=dict(color="#bd9ddf", width=2),
                marker=dict(size=4)
            ))
            fig.add_hline(
                y=target_h,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Target ({target_h:.1f})"
            )
            fig.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis_title="Step",
                yaxis_title="Entropy (bits)"
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        text_placeholder.markdown(full_text)