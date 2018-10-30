class Preprocessor:
  """Text preprocessing algorithm"""
  def __init__(self, text_source):
    pass

  def tokenize(self):
    """text segmentation or lexical analysis"""
    pass

  def normalize(self):
    self.stem()
    self.lemmatize()

  def stem(self):
    """eliminating affixes (suffixed, prefixes,
       infixes, circumfixes) from a word"""
    pass

  def lemmatize(self):
    """determine the canonical form of the words"""
    pass

  def noise_removal(self):
    pass
