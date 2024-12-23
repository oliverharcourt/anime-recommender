class UserNotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class RateLimitExceededError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class InvalidTokenError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
