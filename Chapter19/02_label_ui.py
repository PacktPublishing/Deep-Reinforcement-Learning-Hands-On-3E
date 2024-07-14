#!/usr/bin/env python3
"""
Web interface to label stored data
"""
import argparse

from nicegui import ui
import typing as tt

from lib import ui_tools, rlhf

db: tt.Optional[rlhf.Database] = None
to_label: tt.List[rlhf.HumanLabel] = []



def label_ui():
    with ui.splitter().classes("w-full") as splitter:
        with splitter.before:
            ui.label("List with data samples")
        with splitter.after:
            ui.label("Interface with gif")


@ui.page(ui_tools.URL_ROOT, title="RLHF db overview")
def view_root():
    ui_tools.drawers(ui_tools.URL_ROOT)
    ui.label(f"DB path: {db.db_root}")
    ui.label(f"Trajectories: {len(db.paths)}")
    ui.label(f"Human Labels: {len(db.labels)}")


@ui.page(ui_tools.URL_LABEL, title="RLHF label data")
def view_label():
    ui_tools.drawers(ui_tools.URL_LABEL)
    ui_tools.label_list_view(db, to_label)


@ui.page(ui_tools.URL_DATA, title="RLHF existing data")
def view_label():
    ui_tools.drawers(ui_tools.URL_DATA)
    # make a copy, just in case
    labels_list = list(db.labels)
    ui_tools.label_list_view(db, labels_list, show_resample_list=False)


if __name__ in {"__main__", "__mp_main__"}:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db", required=True, help="DB path to label")
    args = parser.parse_args()

    db = rlhf.load_db(args.db)
    to_label = rlhf.sample_to_label(db)

    ui.run(host='0.0.0.0', port=8080, show=False)
