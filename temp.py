import json
from to_cad import sampleholder_dict_to_cad

# read json file
with open("../data/sampleholder.json", "r") as f:
    sampleholder_dict = json.load(f)
sampleholder_dict_to_cad(sampleholder_dict, cad_file="test.stl")
