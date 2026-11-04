from datetime import datetime, timezone
from functools import lru_cache

from google.cloud import firestore


DAILY_LIMIT = 15
HOURLY_LIMIT = 5


@lru_cache(maxsize=1)
def get_firestore_client():
    return firestore.Client()


def check_usage_limit():
    db = get_firestore_client()

    # Use UTC for consistent dates across instances.
    now = datetime.now(timezone.utc)

    day_id = now.strftime("%Y-%m-%d")
    hour_id = now.strftime("%Y-%m-%d-%H")

    daily_ref = db.collection("daily_usage").document(day_id)
    hourly_ref = db.collection("hourly_usage").document(hour_id)

    transaction = db.transaction()

    @firestore.transactional
    def reserve_request(transaction):
        # Read both counters before writing anything.
        daily_snapshot = daily_ref.get(transaction=transaction)
        hourly_snapshot = hourly_ref.get(transaction=transaction)

        daily_count = (
            daily_snapshot.get("count")
            if daily_snapshot.exists
            else 0
        )

        hourly_count = (
            hourly_snapshot.get("count")
            if hourly_snapshot.exists
            else 0
        )

        if daily_count >= DAILY_LIMIT:
            return False

        if hourly_count >= HOURLY_LIMIT:
            return False

        # Reserve one request in both counters.
        transaction.set(daily_ref, {"count": daily_count + 1})
        transaction.set(hourly_ref, {"count": hourly_count + 1})

        return True

    return reserve_request(transaction)
