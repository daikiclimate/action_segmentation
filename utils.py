def label_to_id(i):
    labels = {
        "opening": 0,
        "moving": 1,
        "hidden": 2,
        "painting": 3,
        "battle": 4,
        "respawn": 5,
        "superjump": 6,
        "object": 7,
        "special": 8,
        "map": 9,
        "ending": 10,
    }
    return labels[i]


def id_to_label(i):
    labels = {
        0: "opening",
        1: "moving",
        2: "hidden",
        3: "painting",
        4: "battle",
        5: "respawn",
        6: "superjump",
        7: "object",
        8: "special",
        9: "map",
        10: "ending",
    }
    return labels[i]

def return_labels():
    return [
        "opening",
        "moving",
        "hidden",
        "painting",
        "battle",
        "respawn",
        "superjump",
        "object",
        "special",
        "map",
        "ending",
        ]

