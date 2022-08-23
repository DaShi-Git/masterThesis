import sys
activation = ["template <typename scalar_t>",
"__device__ __forceinline__ scalar_t arbiacti(scalar_t z) {",
"return z>0.0? z:8.0;//1.0 / (1.0 + exp(-z));",
"}"]
#print("hi")
# class ArbitaryActivation:
#     def __init__(self):
#         print("start write to cuh")

def writeHiddenChannels(activation, truncate):
    cuhFilePath = "arbitaryHiddenChannels.cuh"
    cuhFile = open(cuhFilePath, "w")
    if truncate:
        cuhFile.truncate(0)
    for content in activation:
        bodycontent = ("%s\n" %content)
        cuhFile.write(bodycontent)
    print("written HiddenChannels to cuh file")