import math

class cropper():
    def __init__(self, percent = 0.25):
        self.percentage = percent
        pass
    #https://colab.research.google.com/drive/1oQkPlCREnhiAz0zTifgVi0YWcE6hIzi6#scrollTo=EFMPJVBTNJNh

    def text_fragmeter(self, sentence):

        framented = []
        words = sentence.split(" ")
        if len(words) <= 7:
            return [sentence]
        sen_length = len(words)
        chunk_length = math.ceil(sen_length * self.percentage)

        for i in range(0, sen_length, chunk_length):
            framented.append(" ".join( words[i:i + chunk_length]  ) )

        return framented


            



