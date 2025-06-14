from mmengine.registry import Registry

EVALUATORS = Registry("evaluators")


def build_evaluator(cfg):
    """Build evaluator."""
    return EVALUATORS.build(cfg)


def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        l,s,e = event
        # here, we add removing the events whose duration is 0, (HACS)
        if e - s <= 0:
            continue
        valid = True
        for p_event in valid_events:
            if (
                (abs(s - p_event[1]) <= tol)
                and (abs(e - p_event[2]) <= tol)
                and (l == p_event[0])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events
