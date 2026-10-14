"""Eight synthetic items; Python scalar oracle independent of tensor quantization."""
import torch


def data():
    latents = torch.tensor([[4.9,.1],[5.1,-.1],[4.95,.05],[-3.1,.1],
                            [.1,4.9],[4.7,4.1],[4.3,4.1],[.05,-.05]], dtype=torch.float64)
    books = torch.tensor([[[0,0],[4,0],[0,4],[4,4],[-4,0]],
                          [[0,0],[-1,0],[1,0],[0,-1],[0,1]],
                          [[0,0],[-.25,0],[.25,0],[0,-.25],[0,.25]]], dtype=latents.dtype)
    return latents, books


def scalar_quantize(rows, books):
    codes, reconstructions = [], []
    for row in rows:
        residual, total, selected = list(row), [0.] * len(row), []
        for book in books:
            code = min(range(len(book)), key=lambda i: sum((a-b)**2 for a,b in zip(residual,book[i])))
            selected.append(code)
            total = [a+b for a,b in zip(total,book[code])]
            residual = [a-b for a,b in zip(residual,book[code])]
        codes.append(selected); reconstructions.append(total)
    return codes, reconstructions
