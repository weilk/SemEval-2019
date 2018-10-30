class MLModule:
  """
  Module that encapsulated the machine learning algorithm
  """
  def __init__(self, ml_algorithm):
    self.ml_algorithm = ml_algorithm

  def run(self):
    pass

  def predict(self):
    self.ml_algorithm.predict()
    pass


class AbstractSupervisedMLAlgorithm(object):
  """Abstract class for ML algorithms"""
  def __init__(self):
    pass

  def add_training_data(self, data):
    pass

  def train(self):
    pass

  def test(self):
    pass

  def predict(self):
    pass


class SVM(AbstractSupervisedMLAlgorithm):
  """SVM implementation"""
  def __init__(self):
    super(SVM, self).__init__()

  def train(self):
    pass
    