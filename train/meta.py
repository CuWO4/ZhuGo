__all__ = [
  'MetaData'
]

class MetaData:
  def __init__(self) -> None:
    self.batches: int = 0

  def __str__(self) -> str:
    return f'MetaData{str(self.__dict__)}'