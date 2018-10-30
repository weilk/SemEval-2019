from text_preprocessing import Preprocessor
import machine_learning as ml

class App:
  """App"""
  def __init__(self):
    pass

  def run(self):
    Preprocessor("mocked_text_source")
    ml_module = ml.MLModule(ml.SVM())
    ml_module.run()
    ml_module.predict()


if __name__ == "__main__":
  app = App()
  app.run()
