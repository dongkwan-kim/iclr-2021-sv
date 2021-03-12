
def try_except(func):  # Decorator
    def wrapped(s):
        try:
            return func(s)
        except Exception as e:
            if hasattr(s, "id"):
                print("\nException!", e, s.id)
                return None, e, s.id
            else:
                print("\nException!", e)
                return None, e
    return wrapped
