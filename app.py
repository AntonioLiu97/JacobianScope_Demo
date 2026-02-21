"""
Streamlit app for interactive Jacobian and Temperature Scope visualizations.
"""

import gc
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
import torch
from matplotlib.colors import LogNorm as Log_Norm
from matplotlib.colors import Normalize as Norm
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add current directory to path for JCBScope_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import JCBScope_utils

# Device configuration: use CPU to match notebook and avoid device_map complexity
device = torch.device("cpu")


@st.cache_resource
def load_model(model_name: str = "meta-llama/Llama-3.2-1B"):
    """Load and cache the tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    return tokenizer, model


def check_target_single_token(tokenizer, target_str: str) -> tuple[bool, list[int] | None]:
    """
    Check that target is exactly one token. Returns (ok, ids) or (False, None).
    Uses target_str as-is (no strip) so e.g. " truthful" stays one token.
    """
    ids = tokenizer(target_str, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        return False, None
    return True, ids


def _is_comma_delimited_numbers(s: str) -> bool:
    """Check if string is comma-delimited integers."""
    try:
        parts = [x.strip() for x in s.split(",") if x.strip()]
        return len(parts) > 0 and all(p.lstrip("-").isdigit() for p in parts)
    except Exception:
        return False


def compute_attribution(
    string: str,
    mode: str,
    tokenizer,
    model,
    target_str: str | None = None,
    front_pad: int = 2,
    input_type: str = "text",
):
    """
    Compute attribution using Temperature or Semantic Scope.

    input_type: "text" or "comma_delimited". For comma_delimited, attribution skips delimiter tokens.
    """
    if mode not in ["Temperature", "Semantic"]:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'Temperature' or 'Semantic'.")

    if mode == "Semantic" and (not target_str or not target_str.strip()):
        raise ValueError("Semantic Scope requires a target token.")

    if input_type == "comma_delimited" and not _is_comma_delimited_numbers(string.strip()):
        raise ValueError("Input is not valid comma-delimited numbers.")

    hidden_norm_as_loss = mode == "Temperature"
    back_pad = 0

    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    input_ids_list = []
    if bos_token_id is not None:
        input_ids_list += [bos_token_id] * front_pad
    input_ids_list += tokenizer(string.strip(), add_special_tokens=False)["input_ids"]
    if eos_token_id is not None:
        input_ids_list += [eos_token_id] * back_pad

    embedding_layer = model.get_input_embeddings()
    target_device = embedding_layer.weight.device

    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(target_device)
    attention_mask = torch.ones_like(input_ids)
    assert input_ids.max() < model.config.vocab_size, "Token IDs exceed vocab size"
    assert input_ids.min() >= 0, "Token IDs must be non-negative"

    decoded_tokens = [tokenizer.decode(tok.item(), skip_special_tokens=True) for tok in input_ids[0]]

    if input_type == "comma_delimited":
        grad_idx = list(range(front_pad, len(decoded_tokens), 2))  # Skip delimiter tokens
    else:
        grad_idx = list(range(front_pad, len(decoded_tokens)))

    # loss_position = last position; logits[L-1] predicts the next token after input
    loss_position = len(decoded_tokens) - 1

    target_id = None
    if mode == "Semantic":
        ok, ids = check_target_single_token(tokenizer, target_str)
        if not ok:
            raise ValueError("Target not in token dictionary.")
        target_id = ids[0]

    d_model = embedding_layer.embedding_dim
    residual = nn.Parameter(torch.zeros(len(grad_idx), d_model, device=target_device))
    presence = torch.ones(len(decoded_tokens), 1, device=target_device)

    forward_pass = JCBScope_utils.customize_forward_pass(
        model, residual, presence, input_ids, grad_idx, attention_mask
    )

    unnormalized_logits = True

    loss, logits = forward_pass(
        loss_position=loss_position,
        hidden_norm_as_loss=hidden_norm_as_loss,
        unnormalized_logits=unnormalized_logits,
        tie_input_output_embed=False,
        target_id=target_id,
    )

    grads = torch.autograd.grad(loss, residual, retain_graph=True)[0]

    out = {
        "decoded_tokens": decoded_tokens,
        "grad_idx": grad_idx,
        "grads": grads,
        "loss_position": loss_position,
        "hidden_norm_as_loss": hidden_norm_as_loss,
        "loss": loss.item(),
        "logits": logits,
        "input_type": input_type,
    }
    if mode == "Semantic" and target_str:
        out["target_str"] = target_str  # For visualization: append target in red
    if input_type == "comma_delimited":
        raw = [int(x.strip()) for x in string.strip().split(",") if x.strip()]
        out["int_list"] = raw[: len(grad_idx)]  # align with grad_idx length
    return out


def rgba_to_css(rgba):
    """Convert matplotlib RGBA to CSS rgba string."""
    return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]:.2f})"


def get_text_color(bg_rgba):
    """Return white or black text based on background luminance."""
    luminance = 0.299 * bg_rgba[0] + 0.587 * bg_rgba[1] + 0.114 * bg_rgba[2]
    return "white" if luminance < 0.5 else "black"


def render_attribution_html(result, log_color: bool = False, cmap_name: str = "Blues"):
    """
    Render attribution as HTML with colored token boxes (from notebook routine).
    For Semantic Scope, appends the target token in red.
    """
    decoded_tokens = result["decoded_tokens"]
    grad_idx = result["grad_idx"]
    grads = result["grads"]
    loss_position = result["loss_position"]
    target_str = result.get("target_str")  # Semantic Scope: append target in red
    hardset_target_grad = True
    exclude_target = False

    cmap = plt.get_cmap(cmap_name)

    if exclude_target:
        optimized_tokens = [decoded_tokens[idx] for idx in grad_idx][:-1]
    else:
        optimized_tokens = [decoded_tokens[idx] for idx in grad_idx]

    tick_label_text = optimized_tokens.copy()
    append_target_in_red = target_str is not None

    if len(grads.shape) == 2:
        grad_magnitude = grads.norm(dim=-1).squeeze().detach().clone()
    else:
        grad_magnitude = grads.detach().clone()

    bar_idx = None
    if not exclude_target and hardset_target_grad and (loss_position + 1) in grad_idx:
        target_idx_in_grad = grad_idx.index(loss_position + 1)
        if target_idx_in_grad > 0:
            prev_max = grad_magnitude[:target_idx_in_grad].max().item()
            grad_magnitude[target_idx_in_grad] = max(prev_max, 1e-8)
        else:
            grad_magnitude[target_idx_in_grad] = 1e-8
        bar_idx = target_idx_in_grad

    grad_np = grad_magnitude.cpu().numpy()
    log_norm = Log_Norm(vmin=grad_np.min(), vmax=grad_np.max())
    norm = Norm(vmin=grad_np.min(), vmax=grad_np.max())

    if log_color:
        colors = cmap(log_norm(grad_np))
    else:
        colors = cmap(norm(grad_np))

    html_parts = []
    for i, (token, color) in enumerate(zip(tick_label_text, colors)):
        bg_color = rgba_to_css(color)
        text_color = get_text_color(color)

        if bar_idx is not None and i == bar_idx and hardset_target_grad:
            bg_color = "red"
            text_color = "white"

        display_token = token
        html_parts.append(
            f'<span style="'
            f"background-color: {bg_color}; "
            f"color: {text_color}; "
            f"padding: 0px 0px; "
            f"margin: 0px; "
            f"border-radius: 0px; "
            f"font-family: monospace; "
            f"font-size: 16px; "
            f"display: inline-block; "
            f"font-weight: bold; "
            f'white-space: pre;">{display_token}</span>'
        )

    if append_target_in_red:
        html_parts.append(
            f'<span style="'
            f"background-color: red; "
            f"color: white; "
            f"padding: 0px 0px; "
            f"margin: 0px; "
            f"border-radius: 0px; "
            f"font-family: monospace; "
            f"font-size: 16px; "
            f"display: inline-block; "
            f"font-weight: bold; "
            f'white-space: pre;">{target_str}</span>'
        )

    html_str = f'''
<div style="
    background: white;
    padding: 20px;
    border-radius: 8px;
    line-height: 2.2;
    width: 100%;
    max-width: 700px;
">
    {"".join(html_parts)}
</div>
'''
    # Color bar (from notebook): horizontal, matching the color mapping
    fig_bar, ax_bar = plt.subplots(figsize=(10, 0.3), dpi=100)
    fig_bar.subplots_adjust(left=0.3, right=0.7, bottom=0.1, top=0.9)
    cbar = mpl.colorbar.ColorbarBase(
        ax_bar,
        cmap=cmap,
        norm=log_norm if log_color else norm,
        orientation="horizontal",
    )
    cbar.set_label("Influence")

    return html_str, fig_bar


def render_attribution_barplot(result, log_color: bool = False, cmap_name: str = "Blues"):
    """
    Bar plot with double axes for comma-delimited input: Influence (left) and Token value (right).
    """
    grad_idx = result["grad_idx"]
    grads = result["grads"]
    loss_position = result["loss_position"]
    int_list = result["int_list"]
    front_pad = 2  # assumed

    if len(grads.shape) == 2:
        grad_magnitude = grads.norm(dim=-1).squeeze().detach().clone().cpu().numpy()
    else:
        grad_magnitude = grads.detach().clone().cpu().numpy()

    hardset_target_grad = True
    target_bar_index = None
    if hardset_target_grad and (loss_position + 1) in grad_idx:
        target_bar_index = grad_idx.index(loss_position + 1)
        grad_magnitude[target_bar_index] = max(grad_magnitude)

    ax1_color = np.array([10, 110, 230]) / 256
    ax2_color = np.array([230, 20, 20]) / 256

    x_labels = [x - front_pad for x in grad_idx]

    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=120)
    bars = ax.bar(
        range(grad_magnitude.shape[0]),
        grad_magnitude,
        tick_label=x_labels,
        color=ax1_color,
        linewidth=0.5,
        edgecolor="black",
        width=1.0,
        alpha=0.9,
    )
    if target_bar_index is not None:
        bars[target_bar_index].set_color("red")
        bars[target_bar_index].set_width(1.1)

    ax2 = ax.twinx()
    ax2.scatter(range(len(int_list)), int_list, color=ax2_color, marker="o", s=13, alpha=0.9)
    ax2.plot(range(len(int_list)), int_list, color=ax2_color, linewidth=1.5, alpha=0.5)

    ax2.tick_params(axis="y", colors=ax2_color, labelsize=10)
    ax.tick_params(axis="y", colors=ax1_color, labelsize=10)

    # At most 5 x-axis labels
    n_bars = grad_magnitude.shape[0]
    n_labels = min(5, n_bars)
    if n_labels > 0:
        tick_indices = np.linspace(0, n_bars - 1, n_labels, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([x_labels[i] for i in tick_indices], fontsize=10)

    ax.set_xlabel("Token position index", fontsize=10, fontweight="bold")
    ax.set_ylabel("Influence", labelpad=2, color=ax1_color, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Token value", labelpad=2, color=ax2_color, fontsize=10, fontweight="bold")

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.7)
    ax.yaxis.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.7)

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Jacobian Scope Demo", page_icon="🔬", layout="centered")
    st.title("🔍 Jacobian & Temperature Scopes Demo")
    st.markdown(
        "**Semantic Scope** explains the predicted logit for a specific target token: enter your input "
        "passage along with a target token.\n\n"
        "**Temperature Scope** explains the overall predictive distribution and does not require a target."
    )

    model_choice = st.selectbox(
        "Model",
        options=["LLaMA 3.2 1B", "LLaMA 3.2 3B"],
        index=0,
        key="model_choice",
        help="Choose model.",
    )
    model_name = (
        "meta-llama/Llama-3.2-1B" if model_choice == "LLaMA 3.2 1B" else "meta-llama/Llama-3.2-3B"
    )

    attribution_type = st.radio(
        "Attribution type",
        options=["Semantic Scope", "Temperature Scope"],
        index=0,
        horizontal=True,
        help="Semantic Scope: attribute toward a target token. Temperature Scope: use hidden-state norm.",
    )
    mode = "Semantic" if attribution_type == "Semantic Scope" else "Temperature"

    input_type_default = "text" if mode == "Semantic" else "comma_delimited"
    input_type = st.radio(
        "Input type",
        options=["text", "comma-delimited numbers"],
        index=0 if input_type_default == "text" else 1,
        horizontal=True,
        key=f"input_type_{mode}",
        help="Text: natural language. Comma-delimited numbers: time-series style (delimiters skipped when calculating influence scores).",
    )
    is_comma_delimited = input_type == "comma-delimited numbers"

    if is_comma_delimited:
        default_text = (
            "80,68,57,52,50,49,48,46,42,35,23,14,24,40,49,54,57,60,66,74,79,74,64,58,55,55,57,61,68,77,80,71,60,54,52,51,52,53,55,61,70,83,83,66,53,47,44,41,36,28,22,23,32,40,44,44,43,40,33,24,19,26,37,44,47,47,47,45,40,32,21,16,28,42,49,52,55,58,63,71,80,79,67,58,53,51,51,51,52,55,59,69,82,84,69,54,47,43,40,35,28,22,24,32,39,43,43,41,37,30,22,22,31,39,44,45,44,41,36,27,19,22,34,43,47,49,49,48,47,45,40,31,18,15,31,46,53,57,60,65,72,77,75,67,60,57,57,59,64,71,78,77,68,60,56,55,56,60,66,75,81,75,63,56,53,52,52,54,57,62,73,"
        )
    elif mode == "Semantic":
        default_text = (
            "As a state-of-the-art AI assistant, you never argue or deceive, because you are"
        )
    else:
        default_text = (
            "Italiano: Ma quando tu sarai nel dolce mondo, priegoti ch'a la mente altrui mi rechi: English: But when you have returned to the sweet world, I pray you"
        )

    text_input = st.text_area(
        "Input text",
        value=default_text,
        height=120,
        key=f"text_input_{mode}_{input_type}",
        placeholder="Input text or comma-delimited numbers",
        help="Text or comma-separated numbers. Delimiters are skipped for comma-delimited.",
    )

    target_str = None
    if mode == "Semantic":
        target_str = st.text_input(
            "Target token",
            value=" truthful",
            placeholder='e.g., " truthful" or " nice"',
            help="Must be representable as a single token (e.g. ' truthful' with leading space for Llama).",
        )

    compute_clicked = st.button("Compute Attribution!", type="primary", use_container_width=True)

    input_type_param = "comma_delimited" if is_comma_delimited else "text"

    if compute_clicked:
        if not text_input.strip():
            st.error("Please enter some text.")
        elif mode == "Semantic" and (not target_str or not target_str.strip()):
            st.error("Please enter a target token for Semantic Scope.")
        elif is_comma_delimited and not _is_comma_delimited_numbers(text_input.strip()):
            st.error("Input is not valid comma-delimited numbers.")
        else:
            with st.spinner(f"Loading model and computing {mode} Scope..."):
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect() if torch.cuda.is_available() else None
                    gc.collect()

                    tokenizer, model = load_model(model_name=model_name)
                    result = compute_attribution(
                        text_input.strip(),
                        mode,
                        tokenizer,
                        model,
                        target_str=target_str,
                        input_type=input_type_param,
                    )
                    st.session_state["attribution_result"] = result
                    st.session_state["tokenizer"] = tokenizer

                    st.success("Attribution successful!")
                except ValueError as e:
                    if "Target not in token dictionary" in str(e):
                        st.error("Target not in token dictionary.")
                    else:
                        st.error(str(e))
                except Exception as e:
                    st.error(f"Error: {e}")
                    raise

    # Visualization (uses cached result; log_color and cmap are post-compute only)
    if "attribution_result" in st.session_state:
        result = st.session_state["attribution_result"]
        tokenizer = st.session_state["tokenizer"]

        st.subheader("Attribution Visualization")

        # Adjustable after compute — does not trigger recompute
        viz_col1, viz_col2 = st.columns([1, 1])
        with viz_col1:
            log_color = st.checkbox(
                "Log-scale colormap",
                value=False,
                key="log_color",
                help="Use log scale for influence values.",
            )
        with viz_col2:
            cmap_choice = st.selectbox(
                "Color map",
                options=["Blues", "Greens", "viridis"],
                index=0,
                key="cmap_choice",
                help="Colormap for attribution visualization.",
            )

        if result.get("input_type") == "comma_delimited":
            fig_barplot = render_attribution_barplot(
                result, log_color=log_color, cmap_name=cmap_choice
            )
            st.pyplot(fig_barplot)
            plt.close(fig_barplot)
        else:
            html_output, fig_colorbar = render_attribution_html(
                result, log_color=log_color, cmap_name=cmap_choice
            )
            st.markdown(html_output, unsafe_allow_html=True)
            st.pyplot(fig_colorbar)
            plt.close(fig_colorbar)

        with st.expander("Top predicted next tokens"):
            k = 7
            logit_vector = result["logits"][result["loss_position"]].detach()
            probs = torch.softmax(logit_vector, dim=-1)
            top_probs, top_indices = torch.topk(probs, k)
            top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
            for i, (tok, prob) in enumerate(zip(top_tokens, top_probs.cpu().numpy()), 1):
                st.write(f"{i}. P(**{repr(tok)}**)={prob:.3f}")


if __name__ == "__main__":
    main()
