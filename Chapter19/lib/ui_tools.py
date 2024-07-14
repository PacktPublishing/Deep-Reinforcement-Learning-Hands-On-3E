from nicegui import ui, events
import typing as tt

from lib import rlhf

URL_ROOT = "/"
URL_LABEL = "/label"
URL_DATA = "/data"


DRAWER_ITEMS = (
    ("Overview", URL_ROOT),
    ("Label new data", URL_LABEL),
    ("Existing labels", URL_DATA),
)



def drawers(active_url: str):
    with ui.left_drawer(fixed=True).props("bordered").style('background-color: #ebf1fa') as drawer:
        with ui.column().classes("gap-3 text-body1"):
            for title, link in DRAWER_ITEMS:
                l_obj = ui.link(title, link)
                if link == active_url:
                    l_obj.style("color: red")

    with ui.header(elevated=True):
        with ui.row().classes("fit items-center"):
            ui.button(on_click=lambda: drawer.toggle()).\
                props('flat color=white icon=menu')
            ui.label("RLHF label UI").classes("text-h5 col-grow")


def label_list_view(db: rlhf.Database, items: tt.List[rlhf.HumanLabel],
                    show_resample_list: bool = True):
    selected_row: tt.Optional[dict] = None
    rows = [item.to_json(extra_id=idx) for idx, item in enumerate(items)]

    with ui.row(wrap=False).classes("w-full"):
        with ui.column().classes("w-2/5"):
            if show_resample_list:
                resample_button = ui.button("Resample list").classes("w-full")
            grid = ui.aggrid({
                'columnDefs': [
                    {'headerName': '~', 'field': 'label', 'width': 50},
                    {'headerName': 'Sample 1', 'field': 'sample1'},
                    {'headerName': 'Sample 2', 'field': 'sample2'},
                ],
                'rowSelection': 'single',
                'rowMultiSelectWithClick': True,
                'rowData': rows,
                'domLayout': 'autoHeight',
                ':getRowId': '(params) => params.data.id',
            }).classes("grow")

        def set_label(val: int):
            if selected_row is None:
                return
            idx = selected_row['id']
            items[idx].label = val
            rlhf.store_label(db, items[idx])
            db.labels.append(items[idx])
            grid.options['rowData'][idx] = items[idx].to_json(extra_id=idx)
            grid.update()
            while idx < len(items):
                idx += 1
                if idx == len(items) or items[idx].label is None:
                    break
            if idx < len(items):
                grid.run_row_method(str(idx), "setSelected", True)

        def handle_key(e: events.KeyEventArguments):
            if not e.action.repeat and e.action.keydown:
                if e.key == '1':
                    set_label(1)
                elif e.key == '2':
                    set_label(2)
                elif e.key == '0':
                    set_label(0)

        ui.keyboard(on_key=handle_key)

        with ui.column().classes("w-3/5"):
            with ui.row(wrap=False).classes("w-full"):
                with ui.column().classes("w-1/2"):
                    ui.label("Sample 1").style("font-size: 13pt")
                    img_sample1 = ui.image()
                with ui.column().classes("w-1/2"):
                    ui.label("Sample 2").style("font-size: 13pt")
                    img_sample2 = ui.image()
            with ui.row(wrap=False).classes("w-full"):
                ui.button("#1 is better (1)", on_click=lambda: set_label(1)).classes('w-1/3')
                ui.button("both are good (0)", on_click=lambda: set_label(0)).classes('w-1/3')
                ui.button("#2 is better (2)", on_click=lambda: set_label(2)).classes('w-1/3')
            ui.label("Use buttons above or keys '1', '2' or '0' to set label "
                     "and go to the next sample").style("font-size: 13pt")

    async def _grid_selection_changed(event: events.GenericEventArguments):
        nonlocal selected_row
        selected_row = await grid.get_selected_row()
        print(selected_row)
        s1_path = db.db_root / selected_row['sample1']
        s1_gif = rlhf.get_episode_gif(s1_path)
        img_sample1.set_source(s1_gif)
        s2_path = db.db_root / selected_row['sample2']
        s2_gif = rlhf.get_episode_gif(s2_path)
        img_sample2.set_source(s2_gif)

    first_select = True
    def _grid_data_rendered():
        nonlocal first_select
        if first_select:
            grid.run_row_method("0", "setSelected", True)
        first_select = False

    def _resample_label_list():
        nonlocal rows, first_select
        items.clear()
        sample = rlhf.sample_to_label(db)
        items.extend(sample)
        rows = [item.to_json(extra_id=idx) for idx, item in enumerate(items)]
        first_select = True
        grid.options['rowData'] = rows
        grid.update()

    if show_resample_list:
        resample_button.on_click(_resample_label_list)

    # select first row on data first rendered
    grid.on('firstDataRendered', lambda: _grid_data_rendered())
    grid.on('selectionChanged', _grid_selection_changed)
