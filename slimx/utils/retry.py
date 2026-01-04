import time

def retry(fn, retries=2, base_delay=0.5):
    last=None
    for i in range(retries+1):
        try:
            return fn()
        except Exception as e:
            last=e
            if i>=retries:
                raise
            time.sleep(base_delay*(2**i))
    # If we get here, we never returned successfully.
    # `last` should always be set, but keep the type-checker happy.
    assert last is not None
    raise last
