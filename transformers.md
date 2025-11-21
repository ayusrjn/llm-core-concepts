## Transformers

Transformers are Neural Network Architecture which transformed the Natural Language Processing landscape. Transformers were initially introduced as an improvement to the Machine Translation Architecture in the Paper Attention is All you Need.

### Transformer Architecture

#### Embeddings

Text input is divided into smaller units called tokens, which can be words or subwords. The tokens are converted into numerical vectors called embedding which capture the sematic meaning of the word. Due to these embedding similar words end up nearby For e.g King and Queen end up nearby.

#### Transformer Block

This is the fundamental building block of the model that processes and transforms the input data.

- Attention Layer - Attention Layer is reponsible for generating the attention between the words. It looks into the connection between the words capturing contextual information and relationship between words.

- MLP (Multilayer Perceptron) Layer - This is a feed - forward network that operates on each token independently. While the goal of the attention layer is to figure out information between token, the goal of MLP is to refine each token's representation.

#### Output Probabilities

The final linear normalisation and softmax layer transform the processed embeddings into probabilities, enabling the model to make prediction about the next token in a sequence.

### Embedding

Making sense from text input isn't possible for machines. Text input first has be broken down into tokens. A common misunderstanding is that only words are broken into tokens, but subwords can also be tokens. 
To make the text input, it needs to be converted into a format that the model can understand and process. Embedding turns the text input into numerical representation that model can work with.
The steps followed are 1) Tokenization 2) Obtain Token Embedding 3) add Positional Information 

#### Tokenization


