import json
import pickle
import random
from collections import Counter, defaultdict

from .recommender import Recommender


MAX_ANCHORS = 12
MIN_GOOD_TIME = 0.15
I2I_LOOKAHEAD = 10
RECENCY_DECAY = 0.75
ARTIST_REPEAT_DISCOUNT = 0.9


class SessionWeightedI2IRecommender(Recommender):
    def __init__(self, listen_history_redis, i2i_redis, track_artists, fallback_recommender):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.track_artists = track_artists
        self.fallback_recommender = fallback_recommender

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        seen_tracks = {track for track, _ in history}

        anchor_scores = self._build_anchor_scores(history, prev_track, prev_track_time)
        artist_profile = self._build_artist_profile(history)

        for anchor, _ in anchor_scores[:MAX_ANCHORS]:
            candidate = self._choose_candidate(anchor, seen_tracks, artist_profile)
            if candidate is not None:
                return candidate

        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _build_anchor_scores(self, history, prev_track, prev_track_time):
        scores = defaultdict(float)

        if prev_track_time >= MIN_GOOD_TIME:
            scores[int(prev_track)] += 2.0 * prev_track_time

        decay = 1.0
        for track, listened_time in history:
            engagement = max(float(listened_time), 0.05)
            if listened_time >= MIN_GOOD_TIME:
                scores[int(track)] += engagement * decay
            else:
                scores[int(track)] += 0.15 * engagement * decay
            decay *= RECENCY_DECAY

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _build_artist_profile(self, history):
        counts = Counter()
        for track, listened_time in history:
            artist = self.track_artists.get(int(track))
            if artist is not None:
                counts[artist] += max(float(listened_time), 0.05)
        return counts

    def _choose_candidate(self, anchor: int, seen_tracks, artist_profile):
        raw = self.i2i_redis.get(anchor)
        if raw is None:
            return None

        neighbors = pickle.loads(raw)
        best_track = None
        best_score = None

        for rank, track in enumerate(neighbors[:I2I_LOOKAHEAD]):
            track = int(track)
            if track in seen_tracks:
                continue

            # Higher-ranked I2I neighbors are better.
            rank_score = 1.0 / (1.0 + rank)

            # Avoid over-recommending the same artist too aggressively.
            artist = self.track_artists.get(track)
            artist_penalty = 0.0
            if artist is not None:
                artist_penalty = ARTIST_REPEAT_DISCOUNT * artist_profile.get(artist, 0.0)

            # Tiny random noise helps break ties without changing main logic.
            score = rank_score - artist_penalty + random.random() * 1e-6

            if best_score is None or score > best_score:
                best_score = score
                best_track = track

        return best_track