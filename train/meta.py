__all__ = [
  'MetaData'
]

class MetaData:
  def __init__(self) -> None:
    self.pre1_epochs: int = 0
    self.pre2_epochs: int = 0
    self.epochs: int = 0

  def __str__(self) -> str:
    return f'MetaData{str(self.__dict__)}'