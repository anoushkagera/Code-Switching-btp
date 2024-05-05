import time
import json
import argparse
import torch
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
from src.data.loader import check_all_data_params, load_data
from src.utils import bool_flag, initialize_exp
from src.model import check_mt_model_params, build_mt_model
from src.trainer import TrainerMT
from src.evaluator import EvaluatorMT
from indicnlp.tokenize import indic_tokenize  


def preprocess_sentences(input_hi):
    normalizer = DevanagariNormalizer()
    
    hi_data = []
    
    with open(input_hi, 'r', encoding='utf-8') as hi_reader:
        for line in hi_reader:
            line=line.strip()
            hi_data.append(line)
            
    for i in range(len(hi_data)):
        hi_data[i] = normalizer.normalize(hi_data[i])
    
    return hi_data


def encode_input_sentences(input_hi, encoder_model):
    # Tokenize input sentences
  
    preprocessed_sentences = preprocess_sentences(input_hi)
    
    # Encode input sentences
    encoded_sentences = []
    for sentence in preprocessed_sentences:
        sentence_tensor = torch.LongTensor(sentence).unsqueeze(1)  # Add batch dimension
        if torch.cuda.is_available():
            sentence_tensor = sentence_tensor.cuda()
        encoded_output = encoder_model(sentence_tensor)
        encoded_sentences.append(encoded_output)
    
    return encoded_sentences


def translate_sentence(encoder_output, decoder, lang_id=4, max_length=100):
    """
    Translate a single sentence.
    Args:
        encoder_output: Encoded representation of the input sentence.
        decoder: Decoder model.
        lang_id: Language ID of the target language.
        max_length: Maximum length of the output sentence.
    Returns:
        translated_sentence: Translated sentence.
    """
    # Initialize the decoder input with the encoded representation
    decoder_input = encoder_output.dec_input
    input_length = encoder_output.input_len
    batch_size = encoder_output.input_len.size(0)

    # Initialize an empty tensor to store the decoded sentence
    decoded_words = torch.zeros(max_length, batch_size, dtype=torch.long)
    decoded_words[0] = decoder.bos_index[lang_id]

    # Initialize hidden states
    hidden = None

    # Iterate through each step of decoding
    for t in range(1, max_length):
        # Forward pass through the decoder
        scores = decoder.forward(encoder_output, decoded_words[:t], lang_id)

        # Get the index of the word with the highest score
        top_scores, top_indices = scores[-1].topk(1)

        # Add the word index to the decoded sentence
        decoded_words[t] = top_indices.view(-1)

        # Stop decoding if all sentences have generated an end-of-sentence token
        if (decoded_words[t] == decoder.eos_index).all():
            break

    # Convert the tensor of word indices into a list of words
    translated_sentence = []
    for i in range(batch_size):
        words = [decoder.vocab[token.item()] for token in decoded_words[:, i]]
        # Remove padding and end-of-sentence tokens
        words = [word for word in words if word != '<pad>' and word != '</s>']
        translated_sentence.append(' '.join(words))

    return translated_sentence


def main(params):
    
    ##### can be used for loading encoder and decoder
    # check parameters
    assert params.exp_name
    check_all_data_params(params)
    check_mt_model_params(params)

    # initialize experiment / load data / build model
    logger = initialize_exp(params)
    data = load_data(params)
    encoder, decoder, discriminator, lm = build_mt_model(params, data)

    # initialize trainer / reload checkpoint / initialize evaluator
    trainer = TrainerMT(encoder, decoder, discriminator, lm, data, params)
    trainer.reload_checkpoint()
    trainer.test_sharing()  # check parameters sharing
    evaluator = EvaluatorMT(trainer, data, params)
    
    encoded_sentences = encode_input_sentences(input.txt, trainer.encoder)
    
    translated_data = translate_sentence(encoded_sentences, trainer.decoder, 100)
    
    with open('synthetic_hinglish_sentences', 'w', encoding='utf-8') as f:
        for item in translated_data:
            f.write("%s\n" % item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Hindi to Hinglish")
    args = parser.parse_args()
    main(args)
