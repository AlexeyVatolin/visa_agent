import gradio as gr

from dashboard import compute_tourist_stats

HEADERS = [
    "Country",
    "Avg Wait (days)",
    "Total Cases",
    "Approval Rate (%)",
    "Avg Duration (days)",
    "Multivisas",
]
DATATYPES = ["str", "number", "number", "number", "number", "number"]


def build_dashboard_tab() -> None:
    with gr.Tab("Dashboard"):
        gr.Markdown("# Tourist Visa Dashboard")
        gr.Dataframe(
            headers=HEADERS,
            datatype=DATATYPES,
            value=compute_tourist_stats(),
            interactive=False,
            elem_classes="dashboard-table",
            wrap=True,
        )
