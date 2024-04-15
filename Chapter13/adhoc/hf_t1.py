from transformers import pipeline
import numpy as np

if __name__ == "__main__":
    c = pipeline("feature-extraction")
    r = c(["I'm disappointed by delivery service", "Test sentence"])
    for rr in r:
        a = np.array(rr)
        print(a.shape)