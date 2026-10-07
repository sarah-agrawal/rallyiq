"""Loading ShuttleSet + ShuttleSet22 and turning every shot into model features."""
import glob
import os

import numpy as np
import pandas as pd

SHOT_NAMES = {
    "發短球": "short serve", "發長球": "long serve", "放小球": "net shot",
    "擋小球": "net block", "勾球": "cross-court net", "推球": "push",
    "撲球": "rush", "殺球": "smash", "點扣": "wrist smash", "挑球": "lift",
    "防守回挑": "defensive lift", "長球": "clear", "平球": "drive",
    "小平球": "short drive", "後場抽平球": "back drive", "防守回抽": "defensive drive",
    "切球": "drop", "過度切球": "passive drop", "過渡切球": "passive drop",
    "未知球種": "unknown",
}

# only things known at the moment a shot is hit (no landing spot, no rally result)
PRE_COLS = [
    "ball_round", "hit_x", "hit_y",
    "player_location_x", "player_location_y",
    "opponent_location_x", "opponent_location_y", "opp_dist",
    "hit_height", "aroundhead", "backhand", "frames_since_prev",
    "hitter_score", "opp_score", "score_diff", "game_point",
]

OUTCOMES = ["continues", "winner", "error"]


def clean_name(name):
    return name.lower().replace(" ", "").replace("_", "")


def _load_folder(folder, source):
    frames = []
    for path in glob.glob(os.path.join(folder, "*", "*.csv")):
        df = pd.read_csv(path)
        df["match"] = os.path.basename(os.path.dirname(path))
        df["set"] = os.path.basename(path).replace(".csv", "")
        df["source"] = source
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["match_key"] = out["match"].apply(clean_name)
    return out


def _load_meta(folder):
    match = pd.read_csv(os.path.join(folder, "match.csv"))
    homo = pd.read_csv(os.path.join(folder, "homography.csv"))
    meta = match[["video", "tournament", "round", "year", "winner", "loser"]].merge(
        homo[["video", "upleft_x", "upright_x", "downleft_x", "downright_x",
              "upleft_y", "upright_y", "downleft_y", "downright_y"]], on="video", how="left")
    meta["match_key"] = meta["video"].apply(clean_name)
    return meta


def load_strokes(coachai_dir="CoachAI-Projects"):
    """Return (strokes, matches): one row per shot, one row per match."""
    old_dir = os.path.join(coachai_dir, "ShuttleSet", "set")
    new_dir = os.path.join(coachai_dir, "CoachAI-Challenge-IJCAI2023", "ShuttleSet22", "set")
    old, new = _load_folder(old_dir, "shuttleset"), _load_folder(new_dir, "shuttleset22")

    # a few matches are in both datasets, keep one copy so none leak across train/test
    new = new[~new["match_key"].isin(set(old["match_key"]))]
    strokes = pd.concat([old, new], ignore_index=True)

    meta = pd.concat([_load_meta(old_dir), _load_meta(new_dir)]).drop_duplicates("match_key")
    matches = meta[meta["match_key"].isin(strokes["match_key"])].reset_index(drop=True)

    strokes = strokes.sort_values(["match_key", "set", "rally", "ball_round"]).reset_index(drop=True)
    strokes["rally_id"] = strokes["match_key"] + "_" + strokes["set"] + "_" + strokes["rally"].astype(str)
    strokes["shot"] = strokes["type"].map(SHOT_NAMES).fillna("unknown")

    # who won each rally is only written on the last shot, copy it to every shot
    winners = strokes.dropna(subset=["getpoint_player"]).groupby("rally_id")["getpoint_player"].last()
    strokes["rally_winner"] = strokes["rally_id"].map(winners)
    strokes = strokes[strokes["rally_winner"].notna()].reset_index(drop=True)
    strokes["hitter_wins"] = (strokes["player"] == strokes["rally_winner"]).astype(int)
    return strokes, matches


def add_features(strokes):
    s = strokes.copy()
    g = s.groupby("rally_id")

    # label: what this shot did (0 continues, 1 winner, 2 error)
    s["shots_left"] = g["ball_round"].transform("count") - g.cumcount() - 1
    s["outcome"] = 0
    last = s["shots_left"] == 0
    s.loc[last & (s["hitter_wins"] == 1), "outcome"] = 1
    s.loc[last & (s["hitter_wins"] == 0), "outcome"] = 2

    # score at the START of the rally (the raw columns already include this rally's point)
    rs = s.groupby("rally_id").agg(match_key=("match_key", "first"), set=("set", "first"),
                                   rally=("rally", "first"), end_A=("roundscore_A", "last"),
                                   end_B=("roundscore_B", "last")).sort_values(["match_key", "set", "rally"])
    rs["start_A"] = rs.groupby(["match_key", "set"])["end_A"].shift(1).fillna(0)
    rs["start_B"] = rs.groupby(["match_key", "set"])["end_B"].shift(1).fillna(0)
    s = s.merge(rs[["start_A", "start_B"]], left_on="rally_id", right_index=True, how="left")

    is_a = s["player"] == "A"
    s["hitter_score"] = np.where(is_a, s["start_A"], s["start_B"])
    s["opp_score"] = np.where(is_a, s["start_B"], s["start_A"])
    s["score_diff"] = s["hitter_score"] - s["opp_score"]
    s["game_point"] = ((s["hitter_score"] >= 20) | (s["opp_score"] >= 20)).astype(int)

    g = s.groupby("rally_id")
    s["opp_dist"] = np.hypot(s["player_location_x"] - s["opponent_location_x"],
                             s["player_location_y"] - s["opponent_location_y"])
    s["frames_since_prev"] = g["frame_num"].diff()
    s["prev_shot"] = g["shot"].shift(1).fillna("none")
    s["aroundhead"] = s["aroundhead"].fillna(0)
    s["backhand"] = s["backhand"].fillna(0)
    return s


def build_X(s, columns=None):
    X = pd.concat([
        s[PRE_COLS].astype(float),
        pd.get_dummies(s["shot"], prefix="shot").astype(float),
        pd.get_dummies(s["prev_shot"], prefix="prev").astype(float),
    ], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X
