from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer("sentence-transformers/paraphrase-TinyBERT-L6-v2")


def sentencebert_encode(input_text):
    embedding = model.encode(input_text)

    embedding = torch.tensor(embedding)

    embedding = torch.unsqueeze(embedding, dim=1)

    assert embedding.shape == (len(input_text), 1, 768)

    return embedding


if __name__ == "__main__":
    tensor1 = torch.squeeze(
        sentencebert_encode(["Now is the time to proceed with the go to the drawer action."])
    )

    tensor2 = torch.squeeze(
        sentencebert_encode(["Now is the time to proceed with the go to the cabinet action."])
    )

    tensor3 = torch.squeeze(
        sentencebert_encode(["Why not take hold of the object that's right in front?"])
    )

    print(torch.dot(tensor1, tensor2) / (torch.norm(tensor1) * torch.norm(tensor2)))
