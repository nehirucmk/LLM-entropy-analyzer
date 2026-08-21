import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any

def create_trajectory_plot(telemetry_data: List[Dict[str, Any]], z_threshold: float = 2.0) -> go.Figure:
    if not telemetry_data:
        return go.Figure()

    indices = list(range(len(telemetry_data)))
    tokens = [str(d.get("token", "")).replace("\n", "↵") for d in telemetry_data]
    
    entropies = [float(d["entropy"].item() if hasattr(d.get("entropy", 0.0), "item") else d.get("entropy", 0.0)) for d in telemetry_data]
    z_scores = [float(d["z_score"].item() if hasattr(d.get("z_score", 0.0), "item") else d.get("z_score", 0.0)) for d in telemetry_data]

    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("Shannon Entropy (H) Trajectory", "Moving Z-Score (Anomaly Detection)")
    )

    fig.add_trace(
        go.Scatter(
            x=indices,
            y=entropies,
            mode="lines+markers",
            name="Entropy (H)",
            line=dict(color="#bd9ddf", width=2.5),
            marker=dict(size=5, color="#f472b6"),
            customdata=tokens,
            hovertemplate="<b>Step %{x}</b><br>Token: '%{customdata}'<br>Entropy: %{y:.3f} bits<extra></extra>"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=indices,
            y=z_scores,
            mode="lines+markers",
            name="Z-Score",
            line=dict(color="#38bdf8", width=2),
            marker=dict(
                size=[8 if z >= z_threshold else 4 for z in z_scores],
                color=["#ef4444" if z >= z_threshold else "#38bdf8" for z in z_scores]
            ),
            customdata=tokens,
            hovertemplate="<b>Step %{x}</b><br>Token: '%{customdata}'<br>Z-Score: %{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )

    fig.add_hline(
        y=z_threshold, 
        line_dash="dash", 
        line_color="#ef4444", 
        annotation_text=f"Threshold (Z={z_threshold:.1f})",
        annotation_position="top right",
        annotation_font=dict(color="#ef4444", size=10),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        hovermode="x unified",
        showlegend=False
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
    fig.update_xaxes(title_text="Token Sequence Index", gridcolor="rgba(255,255,255,0.05)", row=2, col=1)
    fig.update_yaxes(title_text="Bits", gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", gridcolor="rgba(255,255,255,0.05)", row=2, col=1)

    return fig


def create_candidate_distribution_plot(candidates: List[Dict[str, Any]]) -> go.Figure:
    if not candidates:
        return go.Figure()

    parsed_candidates = []
    for c in candidates:
        token_text = str(c.get("token", "")).replace("\n", "↵")
        raw_val = c.get("prob", c.get("probability", 0.0))
        if hasattr(raw_val, "item"):
            raw_val = raw_val.item()
            
        try:
            prob_percent = float(raw_val) * 100
        except (ValueError, TypeError):
            prob_percent = 0.0
            
        parsed_candidates.append({"token": token_text, "prob": prob_percent})

    sorted_candidates = sorted(parsed_candidates, key=lambda x: x["prob"])
    token_labels = [c["token"] for c in sorted_candidates]
    probabilities = [c["prob"] for c in sorted_candidates]

    fig = go.Figure(
        go.Bar(
            x=probabilities,
            y=token_labels,
            orientation="h",
            marker=dict(
                color=probabilities,
                colorscale=[[0, "#334155"], [0.5, "#818cf8"], [1, "#bd9ddf"]],
                cmin=0,
                cmax=100,
                line=dict(color="rgba(255,255,255,0.15)", width=1)
            ),
            text=[f"{p:.1f}%" for p in probabilities],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="#ffffff", size=11),
            hovertemplate="Token: '<b>%{y}</b>'<br>Probability: %{x:.2f}%<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(text="Top-K Candidate Probabilities", font=dict(size=12, color="#e2e8f0")),
        template="plotly_dark",
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        xaxis=dict(
            title="Softmax Probability (%)", 
            range=[0, 100], 
            gridcolor="rgba(255,255,255,0.05)"
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
    )

    return fig