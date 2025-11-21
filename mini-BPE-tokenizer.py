import collections
import pickle

class MiniBPE:
	def __init__(self, text_corpus, vocab_size):
		self.text_corpus = text_corpus
		self.tokens = text_corpus.encode("utf-8", errors="replace")
		self.vocab_size = vocab_size
		self.ids = self.train()
		self.vocab = self._build_vocab()


	def _get_stats(self,ids):
		counts = {}
		for pair in zip(ids, ids[1:]):
			counts[pair] = counts.get(pair, 0) + 1

		return counts

	def _merge(self,ids, pair, idx):
		new_index = []
		i = 0
		while i < len(ids):
			if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
				new_index.append(idx)
				i += 2
			else:
				new_index.append(ids[i])
				i += 1

		return new_index

	def train(self):
		num_merges = self.vocab_size - 256
		ids = list(map(int,self.tokens))

		self.merges = {}
		for i in range(num_merges):
			stats = self._get_stats(ids)
			pair = max(stats, key=stats.get)
			idx = 256 + i
			print(f"Merging {pair} into a new token {idx}")
			ids = self._merge(ids, pair, idx)
			self.merges[pair] = idx

		return ids

	def _build_vocab(self):
		vocab = {idx : bytes([idx]) for idx in range(256)}

		for (p0, p1), idx in self.merges.items():
			vocab[idx] = vocab[p0] + vocab[p1]

		return vocab



	def get_compression_stat(self):
		print("Original Token Length : ", len(self.tokens))
		print("Compressed Token Length : ", len(self.ids))
		print(f"Compression Ratio : {len(self.tokens) / len(self.ids):.2f}")
		return

	def decode(self, ids):
		tokens = b"".join(self.vocab[idx] for idx in ids)
		text = tokens.decode("utf-8", errors="replace")

		return text 

	def encode(self, text):
		tokens = list(text.encode("utf-8"))
		while True:
			stats = self._get_stats(tokens)
			pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
			if pair not in self.merges:
				break
			idx = self.merges[pair]
			tokens = self._merge(tokens, pair, idx)
		return tokens

	def save(self, filename):
		model_data = {
			'merges': self.merges,
			'vocab' : self.vocab
		}
		with open(filename, 'wb') as f:
			pickle.dump(model_data, f)

		print(f"Model Saved to {filename}")

	def load(self, filename):
		with open(filename,'rb') as f:
			model_data = pickle.load(f)

		self.merges = model_data['merges']
		self.vocab = model_data['vocab']

		print(f"Model loaded from {filename}")


with open('input.txt', 'r') as i:
	content = str(i.read())

mini_bpe = MiniBPE(content, 500)

mini_bpe.get_compression_stat()
print(mini_bpe.encode("This tokenizer is trained on tiny shakespear"))
mini_bpe.save('tiny_shakespear.pkl')

