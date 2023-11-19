from strictyaml import load, YAMLError, Map, Str
from collections import OrderedDict

try:
    schema = Map({
        "inputDirectory": Str(),
        "inputFile": Str(),
        "outputDirectory": Str(),
        "outputFile": Str()
    })

    with open("./config.yaml", "r") as file:
        configDataMap: OrderedDict = load(yaml_string=file.read(), schema=schema).data
    file.close()

except YAMLError as error:
    print(error)
    exit()


class ConfigReader:
    def __init__(self):
        self.inputDirectory = configDataMap.get("inputDirectory")
        self.inputFile = configDataMap.get("inputFile")
        self.outputDirectory = configDataMap.get("outputDirectory")
        self.outputFile = configDataMap.get("outputFile")
