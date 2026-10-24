"""Hand-calculation formulas (Ch. 4). FLOPs = 2 x MACs; biases, activations, norms, softmax ignored.

These functions reproduce what you write on paper; cost.py compares them with thop.
"""

def linear(inp, out, tokens=1):
    return 2 * inp * out * tokens


def conv2d(k, cin, cout, hout, wout):
    return 2 * k * k * cin * cout * hout * wout


def attention_layer(seq=16, d=64, ff=128):
    parts = {
        "QKV projections": 2 * seq * d * 3 * d,
        "scores QK^T": 2 * seq * seq * d,
        "weights x V": 2 * seq * seq * d,
        "output projection": 2 * seq * d * d,
        "FFN linear 1": 2 * seq * d * ff,
        "FFN linear 2": 2 * seq * ff * d,
    }
    return parts


def model_totals():
    mlp = [linear(784, 256), linear(256, 128), linear(128, 10)]
    cnn = [conv2d(3, 1, 16, 28, 28), conv2d(3, 16, 32, 14, 14), linear(32 * 7 * 7, 10)]
    layer = sum(attention_layer().values())
    att = [linear(49, 64, tokens=16), layer, layer, linear(64, 10)]
    return {"MLP": sum(mlp), "CNN": sum(cnn), "Attention": sum(att)}


if __name__ == "__main__":
    print("one layer each: Linear 784->256 =", linear(784, 256),
          "| Conv2d(1,16,3) @28x28 =", conv2d(3, 1, 16, 28, 28))
    print("attention layer parts:", attention_layer(), "sum =", sum(attention_layer().values()))
    print("model totals:", model_totals())
