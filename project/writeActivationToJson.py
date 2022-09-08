import json
activation = {"ReLU":[\
    "template <typename scalar_t>", \
    "__device__ __forceinline__ scalar_t arbiacti1(scalar_t z) {", \
    "return z>(half)0.0? z:(half)0.0;", \
    "}"]}
# activation = json.dumps(activation)
# json_dict = json.loads(activation)
with open('activationLibrary.json', 'r') as json_file:
    #json.dump(activation, json_file)
    data = json.load(json_file)
print(data["ReLU"])

