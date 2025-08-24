import ipywidgets as widgets
from IPython.display import clear_output, display

from class_factory.utils.tools import normalize_unicode


def prompt_user_for_reading_matches(lesson_index, match_threshold=0.8, top_n=3, on_submit_callback=None):
    """
    lesson_index: dict in the new lesson-centric format
    match_threshold: float, below which a reading is considered unmatched
    top_n: int, number of top matches to show
    on_submit_callback: function(selections, lesson_index) called after submit
    Returns: dict {reading: selected_file or None}
    """
    # Extract unmatched readings and top matches from lesson_index
    unmatched_rows = []  # List of (lesson, reading, matches, assigned_file)
    for lesson, lesson_entry in lesson_index.items():
        for reading, entry in lesson_entry["readings"].items():
            matches = entry["matches"]
            # Use 'score' if 'combined_score' is not present
            score_key = "combined_score" if matches and "combined_score" in matches[0] else "score"
            assigned_file = entry.get("assigned_file")
            if not matches or matches[0][score_key] < match_threshold or assigned_file is None:
                unmatched_rows.append((lesson, reading, matches[:top_n], assigned_file))

    selections = {}
    dropdowns = []
    rows = []
    # Column headers
    header = widgets.HBox([
        widgets.HTML('<b>Lesson</b>', layout=widgets.Layout(width='10%')),
        widgets.HTML('<b>Syllabus Reading</b>', layout=widgets.Layout(width='55%')),
        widgets.HTML('<b>Possible Matches</b>', layout=widgets.Layout(width='35%'))
    ])
    for lesson, reading, matches, assigned_file in unmatched_rows:
        clean_reading = normalize_unicode(reading)
        options = []
        for m in matches:
            clean_snippet = normalize_unicode(m["matched_text"])
            label = f'{m["file"]} (score: {m["combined_score"]}) - {clean_snippet[:40]}'
            options.append((label, m["file"]))
        options.append(('Skip/None', None))
        options.append(('Not a reading', '__NOT_A_READING__'))
        # Set value to assigned_file if present, otherwise None
        value = assigned_file if assigned_file in [m["file"] for m in matches] else None
        if assigned_file == '__NOT_A_READING__':
            value = '__NOT_A_READING__'
        dropdown = widgets.Dropdown(
            options=options,
            value=value,
            layout=widgets.Layout(width='98%')
        )
        dropdowns.append(dropdown)
        # Lesson number (left column)
        lesson_col = widgets.HTML(f'<div>{lesson}</div>', layout=widgets.Layout(width='10%'))
        # Syllabus reading text (middle column)
        left = widgets.HTML(f'<div style="white-space:pre-wrap">{clean_reading}</div>', layout=widgets.Layout(width='55%'))
        # Dropdown (right column, narrower)
        right = widgets.Box([dropdown], layout=widgets.Layout(width='35%'))
        row = widgets.HBox([lesson_col, left, right])
        rows.append(row)

    submit_button = widgets.Button(description="Submit Selections", button_style='success')
    output = widgets.Output()

    def on_submit(b):
        with output:
            clear_output()
            for (_, reading, _, _), dropdown in zip(unmatched_rows, dropdowns):
                selections[reading] = dropdown.value
            # Update lesson_index with new assignments
            for lesson, reading, matches, _ in unmatched_rows:
                assigned = selections[reading]
                lesson_entry = lesson_index[lesson]["readings"][reading]
                lesson_entry["assigned_file"] = assigned
            print("Selections submitted and assignments saved (in-memory). If you want to persist to disk, save lesson_index after this step.")
            # Order selections by file (None at the end, Not a reading at the end)
            ordered = {}
            files = sorted({v for v in selections.values() if v not in (None, '__NOT_A_READING__')})
            for file in files:
                ordered[file] = [k for k, v in selections.items() if v == file]
            not_a_reading = [k for k, v in selections.items() if v == '__NOT_A_READING__']
            unassigned = [k for k, v in selections.items() if v is None]
            if not_a_reading:
                ordered['Not a reading'] = not_a_reading
            if unassigned:
                ordered[None] = unassigned
            display(ordered)
            if on_submit_callback is not None:
                on_submit_callback(selections, lesson_index)

    submit_button.on_click(on_submit)
    display(widgets.VBox([header] + rows + [submit_button, output]))
    # No blocking wait; result is handled via callback
    return None


def prompt_user_assign_unmatched(unmatched_readings, lessons, expected_counts, current_assignments=None, on_submit_callback=None):
    """
    Display a GUI for assigning unmatched readings to lessons, showing expected counts.
    unmatched_readings: list of reading file names
    lessons: list of lesson numbers
    expected_counts: dict {lesson_no: expected_count}
    current_assignments: dict {lesson_no: [reading1, ...]} (optional)
    Returns: dict {lesson_no: [reading1, ...]}
    """
    if current_assignments is None:
        current_assignments = {lesson: [] for lesson in lessons}
    else:
        for lesson in lessons:
            current_assignments.setdefault(lesson, [])

    # Only show unassigned readings, each with a multi-select for lessons
    select_widgets = {}
    rows = []
    header = widgets.HBox([
        widgets.HTML('<b>Unassigned Reading</b>', layout=widgets.Layout(width='60%')),
        widgets.HTML('<b>Assign to Lesson(s)</b>', layout=widgets.Layout(width='40%'))
    ])
    if unmatched_readings:
        reading_widgets = []
        for reading in sorted(unmatched_readings):
            select = widgets.SelectMultiple(
                options=lessons,
                value=(),
                layout=widgets.Layout(width='80%')
            )
            select_widgets[reading] = select
            reading_row = widgets.HBox([
                widgets.HTML(f'<span style="white-space:pre-wrap">{reading}</span>', layout=widgets.Layout(width='60%')),
                select
            ])
            reading_widgets.append(reading_row)
        rows.append(widgets.VBox(reading_widgets))
    else:
        rows.append(widgets.HTML('<i>No unassigned readings found.</i>'))

    submit_button = widgets.Button(description="Submit Assignments", button_style='success')
    output = widgets.Output()

    def on_submit(b):
        with output:
            clear_output()
            # Build new assignments: {lesson: [reading, ...]}
            new_assignments = {lesson: [] for lesson in lessons}
            for reading, select in select_widgets.items():
                for lesson in select.value:
                    new_assignments[lesson].append(reading)
            print("Assignments submitted!")
            display(new_assignments)
            if on_submit_callback is not None:
                # Pass new_assignments and the full select_widgets for possible index update
                on_submit_callback(new_assignments, None)

    submit_button.on_click(on_submit)
    display(widgets.VBox([header] + rows + [submit_button, output]))
    # No blocking wait; result is handled via callback
    return None
