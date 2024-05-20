class LLMException(BaseException):
    def __init__(self, type, prompt, response):
        super(LLMException, self).__init__()
        
        self.type = type
        self.prompt = prompt
        self.response = response
    
    def __str__(self):
        str = "+++++ LLM Exception +++++\n"
        str += "[Type]: %s\n" % self.type
        str += "[Prompt]: %s\n" % self.prompt
        str += "[Response]: %s\n" % self.response
        str += "++++++++++++++++++++++++\n"
        return str

class MechanismException(BaseException):
    def __init__(self, type, message):
        super(MechanismException, self).__init__()

        self.type = type
        self.message = message

    def __str__(self):
        str = "+++++ LLM Exception +++++\n"
        str += "[Type]: %s\n" % self.type
        str += "[Message]: %s\n" % self.message
        str += "++++++++++++++++++++++++\n"
        return str