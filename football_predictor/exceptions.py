class FootballPredictorError(Exception):
    """Base application exception safe to show in the UI."""


class ValidationError(FootballPredictorError):
    pass


class DuplicateMatchError(FootballPredictorError):
    pass


class InsufficientDataError(FootballPredictorError):
    pass


class ModelNotTrainedError(FootballPredictorError):
    pass


class ModelTrainingError(FootballPredictorError):
    pass


class OCRUnavailableError(FootballPredictorError):
    pass


class ScreenshotParseError(FootballPredictorError):
    pass

