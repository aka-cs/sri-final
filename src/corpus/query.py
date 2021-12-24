class Query:
    
    def __init__(self, text: str):
        self.text = text
        
    def __iter__(self):
        for w in self.text.split():
            yield w
